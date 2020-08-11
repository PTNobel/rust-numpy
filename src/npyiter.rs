use crate::array::{PyArray, PyArrayDyn};
use crate::npyffi::{
    array::PY_ARRAY_API,
    types::{NPY_CASTING, NPY_ORDER},
    *,
};
use crate::types::Element;
use crate::error::NpyIterInstantiationError;
use pyo3::{prelude::*, PyNativeType};

use std::marker::PhantomData;
use std::os::raw::*;
use std::ptr;

/// Flags for `NpySingleIter` and `NpyMultiIter`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NpyIterFlag {
    /* CIndex,
    FIndex,
    MultiIndex, */
    // ExternalLoop, // This flag greatly modifies the behaviour of accessing the data
    // so we don't support it.
    CommonDtype,
    RefsOk,
    ZerosizeOk,
    ReduceOk,
    Ranged,
    Buffered,
    GrowInner,
    DelayBufAlloc,
    DontNegateStrides,
    CopyIfOverlap,
    /* ReadWrite,
    ReadOnly,
    WriteOnly, */
}

impl NpyIterFlag {
    fn to_c_enum(&self) -> npy_uint32 {
        use NpyIterFlag::*;
        match self {
            /* CIndex => NPY_ITER_C_INDEX,
            FIndex => NPY_ITER_C_INDEX,
            MultiIndex => NPY_ITER_MULTI_INDEX, */
            /* ExternalLoop => NPY_ITER_EXTERNAL_LOOP, */
            CommonDtype => NPY_ITER_COMMON_DTYPE,
            RefsOk => NPY_ITER_REFS_OK,
            ZerosizeOk => NPY_ITER_ZEROSIZE_OK,
            ReduceOk => NPY_ITER_REDUCE_OK,
            Ranged => NPY_ITER_RANGED,
            Buffered => NPY_ITER_BUFFERED,
            GrowInner => NPY_ITER_GROWINNER,
            DelayBufAlloc => NPY_ITER_DELAY_BUFALLOC,
            DontNegateStrides => NPY_ITER_DONT_NEGATE_STRIDES,
            CopyIfOverlap => NPY_ITER_COPY_IF_OVERLAP,
            /* ReadWrite => NPY_ITER_READWRITE,
            ReadOnly => NPY_ITER_READONLY,
            WriteOnly => NPY_ITER_WRITEONLY, */
        }
    }
}

pub struct NpySingleIterBuilder<'py, T> {
    flags: npy_uint32,
    array: &'py PyArrayDyn<T>,
}

impl<'py, T: Element> NpySingleIterBuilder<'py, T> {
    pub fn readwrite<D: ndarray::Dimension>(array: &'py PyArray<T, D>) -> Self {
        Self {
            flags: NPY_ITER_READWRITE,
            array: array.to_dyn(),
        }
    }

    pub fn readonly<D: ndarray::Dimension>(array: &'py PyArray<T, D>) -> Self {
        Self {
            flags: NPY_ITER_READONLY,
            array: array.to_dyn(),
        }
    }

    pub fn set(mut self, flag: NpyIterFlag) -> Self {
        self.flags |= flag.to_c_enum();
        self
    }

    pub fn unset(mut self, flag: NpyIterFlag) -> Self {
        self.flags &= !flag.to_c_enum();
        self
    }

    pub fn build(self) -> PyResult<NpySingleIter<'py, T>> {
        let iter_ptr = unsafe {
            PY_ARRAY_API.NpyIter_New(
                self.array.as_array_ptr(),
                self.flags,
                NPY_ORDER::NPY_ANYORDER,
                NPY_CASTING::NPY_SAFE_CASTING,
                ptr::null_mut(),
            )
        };
        let py = self.array.py();
        NpySingleIter::new(iter_ptr, py)
    }
}

pub struct NpySingleIter<'py, T> {
    iterator: ptr::NonNull<objects::NpyIter>,
    iternext: unsafe extern "C" fn(*mut objects::NpyIter) -> c_int,
    empty: bool,
    iter_size: npy_intp,
    dataptr: *mut *mut c_char,
    return_type: PhantomData<T>,
    _py: Python<'py>,
}

impl<'py, T> NpySingleIter<'py, T> {
    fn new(iterator: *mut objects::NpyIter, py: Python<'py>) -> PyResult<Self> {
        let mut iterator = match ptr::NonNull::new(iterator) {
            Some(iter) => iter,
            None => {
                return Err(NpyIterInstantiationError.into());
            }
        };

        let iternext = match unsafe { PY_ARRAY_API.NpyIter_GetIterNext(iterator.as_mut(), ptr::null_mut()) } {
            Some(ptr) => ptr,
            None => {
                return Err(PyErr::fetch(py));
            }
        };
        let dataptr = unsafe { PY_ARRAY_API.NpyIter_GetDataPtrArray(iterator.as_mut()) };

        if dataptr.is_null() {
            unsafe { PY_ARRAY_API.NpyIter_Deallocate(iterator.as_mut()) };
            return Err(NpyIterInstantiationError.into());
        }

        let iter_size = unsafe { PY_ARRAY_API.NpyIter_GetIterSize(iterator.as_mut()) };

        Ok(Self {
            iterator,
            iternext,
            iter_size,
            empty: iter_size == 0,
            dataptr,
            return_type: PhantomData,
            _py: py,
        })
    }
}

impl<'py, T> Drop for NpySingleIter<'py, T> {
    fn drop(&mut self) {
        let _success = unsafe { PY_ARRAY_API.NpyIter_Deallocate(self.iterator.as_mut()) };
        // TODO: Handle _success somehow?
    }
}

impl<'py, T: 'py> std::iter::Iterator for NpySingleIter<'py, T> {
    type Item = &'py T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.empty {
            None
        } else {
            // Note: This pointer is correct and doesn't need to be updated,
            // note that we're derefencing a **char into a *char casting to a *T
            // and then transforming that into a reference, the value that dataptr
            // points to is being updated by iternext to point to the next value.
            let retval = Some(unsafe { &*(*self.dataptr as *mut T) });
            self.empty = unsafe { (self.iternext)(self.iterator.as_mut()) } == 0;
            retval
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.iter_size as usize, Some(self.iter_size as usize))
    }
}

mod private {
    pub struct PrivateGuard;
}
macro_rules! private_decl {
    () => {
        fn __private__() -> private::PrivateGuard;
    };
}
macro_rules! private_impl {
    () => {
        fn __private__() -> private::PrivateGuard {
            private::PrivateGuard
        }
    };
}

/// A combinator type that represents an terator mode (e.g., ReadOnly + ReadWrite).
pub trait MultiIterMode {
    private_decl!();
    type Pre: MultiIterMode;
    const FLAG: npy_uint32 = 0;
    fn flags() -> Vec<npy_uint32> {
        if Self::FLAG == 0 {
            vec![]
        } else {
            let mut res = Self::Pre::flags();
            res.push(Self::FLAG);
            res
        }
    }
}

impl MultiIterMode for () {
    private_impl!();
    type Pre = ();
}

// https://stackoverflow.com/questions/57454887/how-do-i-append-to-a-tuple
pub trait TupleAppend<T> {
    private_decl!();
    type ResultType;

    fn append(self, t: T) -> Self::ResultType;
}

impl<T> TupleAppend<T> for () {
    private_impl!();
    type ResultType = (T,);

    fn append(self, t: T) -> Self::ResultType {
        (t,)
    }
}

macro_rules! impl_tuple_append {
    ( () ) => {};
    ( ( $t0:ident $(, $types:ident)* ) ) => {
        impl<$t0, $($types,)* T> TupleAppend<T> for ($t0, $($types,)*) {
            private_impl!();
            // Trailing comma, just to be extra sure we are dealing
            // with a tuple and not a parenthesized type/expr.
            type ResultType = ($t0, $($types,)* T,);

            fn append(self, t: T) -> Self::ResultType {
                // Reuse the type identifiers to destructure ourselves:
                let ($t0, $($types,)*) = self;
                // Create a new tuple with the original elements, plus the new one:
                ($t0, $($types,)* t,)
            }
        }

        // Recurse for one smaller size:
        impl_tuple_append! { ($($types),*) }
    };
}

impl_tuple_append! {
    // Supports tuples up to size 10:
    (_1, _2, _3, _4, _5, _6, _7, _8, _9, _10)
}

/// A builder struct for creating multi iterator.
pub struct NpyMultiIterBuilder<'py, T, S> {
    flags: npy_uint32,
    opflags: Vec<npy_uint32>,
    arrays: Vec<&'py PyArrayDyn<T>>,
    structure: PhantomData<S>,
}

impl<'py, T: Element> NpyMultiIterBuilder<'py, T, ()> {
    pub fn new() -> Self {
        Self {
            flags: 0,
            opflags: Vec::new(),
            arrays: Vec::new(),
            structure: PhantomData,
        }
    }

    pub fn set(mut self, flag: NpyIterFlag) -> Self {
        self.flags |= flag.to_c_enum();
        self
    }

    pub fn unset(mut self, flag: NpyIterFlag) -> Self {
        self.flags &= !flag.to_c_enum();
        self
    }
}

impl<'py, T: Element, S: TupleAppend<&'py T>> NpyMultiIterBuilder<'py, T, S> {
    pub fn add_readonly_array<D: ndarray::Dimension>(
        mut self,
        array: &'py PyArray<T, D>,
    ) -> NpyMultiIterBuilder<'py, T, S::ResultType> {
        self.arrays.push(array.to_dyn());
        self.opflags.push(NPY_ITER_READONLY);

        NpyMultiIterBuilder {
            flags: self.flags,
            opflags: self.opflags,
            arrays: self.arrays,
            structure: PhantomData,
        }
    }
}

impl<'py, T: Element, S: TupleAppend<&'py mut T>> NpyMultiIterBuilder<'py, T, S> {
    pub fn add_readwrite_array<D: ndarray::Dimension>(
        mut self,
        array: &'py PyArray<T, D>,
    ) -> NpyMultiIterBuilder<'py, T, S::ResultType> {
        self.arrays.push(array.to_dyn());
        self.opflags.push(NPY_ITER_READWRITE);

        NpyMultiIterBuilder {
            flags: self.flags,
            opflags: self.opflags,
            arrays: self.arrays,
            structure: PhantomData,
        }
    }
}

impl<'py, T: Element, S> NpyMultiIterBuilder<'py, T, S> {
    pub fn build(mut self) -> PyResult<NpyMultiIter<'py, T, S>> {
        assert!(self.arrays.len() <= i32::MAX as usize);
        assert!(2 <= self.arrays.len());

        let iter_ptr = unsafe {
            PY_ARRAY_API.NpyIter_MultiNew(
                self.arrays.len() as i32,
                self.arrays
                    .iter_mut()
                    .map(|x| x.as_array_ptr())
                    .collect::<Vec<_>>()
                    .as_mut_ptr(),
                self.flags,
                NPY_ORDER::NPY_ANYORDER,
                NPY_CASTING::NPY_SAFE_CASTING,
                self.opflags.as_mut_ptr(),
                ptr::null_mut(),
            )
        };
        let py = self.arrays[0].py();
        NpyMultiIter::new(iter_ptr, py)
    }
}

/// Multi iterator
pub struct NpyMultiIter<'py, T, S> {
    iterator: ptr::NonNull<objects::NpyIter>,
    iternext: unsafe extern "C" fn(*mut objects::NpyIter) -> c_int,
    empty: bool,
    iter_size: npy_intp,
    dataptr: *mut *mut c_char,
    marker: PhantomData<(T, S)>,
    _py: Python<'py>,
}

impl<'py, T, S> NpyMultiIter<'py, T, S> {
    fn new(iterator: *mut objects::NpyIter, py: Python<'py>) -> PyResult<Self> {
        let mut iterator = match ptr::NonNull::new(iterator) {
            Some(ptr) => ptr,
            None => {
                return Err(NpyIterInstantiationError.into());
            }
        };

        let iternext = match unsafe { PY_ARRAY_API.NpyIter_GetIterNext(iterator.as_mut(), ptr::null_mut()) } {
            Some(ptr) => ptr,
            None => {
                return Err(PyErr::fetch(py));
            }
        };
        let dataptr = unsafe { PY_ARRAY_API.NpyIter_GetDataPtrArray(iterator.as_mut()) };

        if dataptr.is_null() {
            unsafe { PY_ARRAY_API.NpyIter_Deallocate(iterator.as_mut()) };
            return Err(NpyIterInstantiationError.into());
        }

        let iter_size = unsafe { PY_ARRAY_API.NpyIter_GetIterSize(iterator.as_mut()) };

        Ok(Self {
            iterator,
            iternext,
            iter_size,
            empty: iter_size == 0,
            dataptr,
            marker: PhantomData,
            _py: py,
        })
    }
}



impl<'py, T, S> Drop for NpyMultiIter<'py, T, S> {
    fn drop(&mut self) {
        let _success = unsafe { PY_ARRAY_API.NpyIter_Deallocate(self.iterator.as_mut()) };
        // TODO: Handle _success somehow?
    }
}

macro_rules! impl_multi_iter {
    ($expand: ident, $deref: expr) => {
        impl <'py, T: 'py, S: TupleAppend<T>> std::iter::Iterator for NpyMultiIter<'py, T, S> {
            type Item = S;

            fn next(&mut self) -> Option<Self::Item> {
                if self.empty {
                    None
                } else {
                    // Note: This pointer is correct and doesn't need to be updated,
                    // note that we're derefencing a **char into a *char casting to a *T
                    // and then transforming that into a reference, the value that dataptr
                    // points to is being updated by iternext to point to the next value.
                    // let ($($ptr,)+) = unsafe { $expand::<T>(self.dataptr) };
                    let $expand = self.dataptr;
                    let retval = Some(unsafe { $deref });
                    self.empty = unsafe { (self.iternext)(self.iterator.as_mut()) } == 0;
                    retval
                }
            }

            fn size_hint(&self) -> (usize, Option<usize>) {
                (self.iter_size as usize, Some(self.iter_size as usize))
            }
        }
    }
}

impl_multi_iter!(ptr, ((*ptr).cast(),));

/* macro_rules! impl_multi_iter {
    ($structure: ty, $($ty: ty)+, $($ptr: ident)+, $expand: ident, $deref: expr) => {
        impl<'py, T: 'py> std::iter::Iterator for NpyMultiIter<'py, T, $structure> {
            type Item = ($($ty,)+);
            fn next(&mut self) -> Option<Self::Item> {
                if self.empty {
                    None
                } else {
                    // Note: This pointer is correct and doesn't need to be updated,
                    // note that we're derefencing a **char into a *char casting to a *T
                    // and then transforming that into a reference, the value that dataptr
                    // points to is being updated by iternext to point to the next value.
                    let ($($ptr,)+) = unsafe { $expand::<T>(self.dataptr) };
                    let retval = Some(unsafe { $deref });
                    self.empty = unsafe { (self.iternext)(self.iterator.as_mut()) } == 0;
                    retval
                }
            }

            fn size_hint(&self) -> (usize, Option<usize>) {
                (self.iter_size as usize, Some(self.iter_size as usize))
            }
        }
    };
}
*/
