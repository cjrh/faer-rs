//! Regression for callers that must handle allocation failure before kernels
//! run.
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

thread_local! {
	static DENY: Cell<bool> = const { Cell::new(false) };
}
struct Allocator;
#[global_allocator]
static ALLOCATOR: Allocator = Allocator;

unsafe impl GlobalAlloc for Allocator {
	unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
		if DENY.with(Cell::get) {
			core::ptr::null_mut()
		} else {
			unsafe { System.alloc(layout) }
		}
	}

	unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
		if DENY.with(Cell::get) {
			core::ptr::null_mut()
		} else {
			unsafe { System.alloc_zeroed(layout) }
		}
	}

	unsafe fn realloc(
		&self,
		ptr: *mut u8,
		layout: Layout,
		size: usize,
	) -> *mut u8 {
		if DENY.with(Cell::get) {
			core::ptr::null_mut()
		} else {
			unsafe { System.realloc(ptr, layout, size) }
		}
	}

	unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
		unsafe { System.dealloc(ptr, layout) }
	}
}

#[test]
fn cold_failure_then_reserved_sequential_kernels() {
	// A fresh thread ensures that a prior test cannot hide the initial
	// allocation.
	std::thread::spawn(|| {
		DENY.with(|deny| deny.set(true));
		let cold = faer::linalg::matmul::GemmWorkspace::try_new();
		DENY.with(|deny| deny.set(false));
		#[cfg(target_arch = "x86_64")]
		if std::is_x86_feature_detected!("avx512f")
			|| (std::is_x86_feature_detected!("avx2")
				&& std::is_x86_feature_detected!("fma"))
		{
			assert!(cold.is_err());
		}
		let mut workspace =
			faer::linalg::matmul::GemmWorkspace::try_new().unwrap();
		macro_rules! check {
			($ty:ty, $one:expr, $answer:expr) => {{
				let n = 65;
				let a = vec![$one; n * n];
				let mut c = vec![<$ty>::default(); n * n];
				let a = faer::MatRef::from_column_major_slice(&a, n, n);
				DENY.with(|deny| deny.set(true));
				workspace.with(|| {
					faer::linalg::matmul::matmul(
						faer::MatMut::from_column_major_slice_mut(&mut c, n, n),
						faer::Accum::Replace,
						a,
						a,
						$one,
						faer::Par::Seq,
					)
				});
				DENY.with(|deny| deny.set(false));
				assert!(c.iter().all(|&value| value == $answer));
			}};
		}
		check!(f32, 1.0f32, 65.0f32);
		check!(f64, 1.0f64, 65.0f64);
		check!(
			faer::c32,
			faer::c32::new(1.0, 0.0),
			faer::c32::new(65.0, 0.0)
		);
		check!(
			faer::c64,
			faer::c64::new(1.0, 0.0),
			faer::c64::new(65.0, 0.0)
		);
	})
	.join()
	.unwrap();
}

#[test]
fn nested_scope_unwinding_restores_the_outer_workspace() {
	let mut outer = faer::linalg::matmul::GemmWorkspace::try_new().unwrap();
	let mut inner = faer::linalg::matmul::GemmWorkspace::try_new().unwrap();
	let n = 65;
	let a = vec![1.0f64; n * n];
	let mut c = vec![0.0f64; n * n];
	outer.with(|| {
		let unwind =
			std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
				inner.with(|| panic!("exercise scope cleanup"));
			}));
		assert!(unwind.is_err());
		drop(inner);
		let a = faer::MatRef::from_column_major_slice(&a, n, n);
		DENY.with(|deny| deny.set(true));
		faer::linalg::matmul::matmul(
			faer::MatMut::from_column_major_slice_mut(&mut c, n, n),
			faer::Accum::Replace,
			a,
			a,
			1.0,
			faer::Par::Seq,
		);
		DENY.with(|deny| deny.set(false));
	});
	assert!(c.iter().all(|&value| value == n as f64));
}
