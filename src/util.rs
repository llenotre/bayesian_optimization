use std::f64::consts::PI;

use leonhard::linear_algebra::*;

static mut RAND: u64 = 42;

fn pseudo_rand() -> f64 {
	let a: u64 = 25214903917;
	let c: u64 = 11;
	let m: u64 = 1 << 45;

	unsafe {
		let x = (a.wrapping_mul(RAND).wrapping_add(c)) % m;
		RAND = x;
		(x as f64) / (m as f64)
	}
}

pub fn rand_vector(size: usize, amplitude: f64) -> Vector<f64> {
	let mut vec = Vec::<f64>::new();
	for _ in 0..size {
		vec.push((pseudo_rand() - 0.5) * (amplitude * 2.));
	}
	Vector::<f64>::from_vec(vec)
}

pub fn maxf(a: f64, b:  f64) -> f64 {
	if a > b {
		a
	} else {
		b
	}
}

pub fn erf(x: f64) -> f64 {
	if x >= 3. {
		return 1.;
	} else if x <= -3. {
		return -1.;
	}

	let mut r0 = 0.;
	for n in 0..100 {
		let mut r1 = 1.;
		for k in 1..=n {
			r1 *= -(x * x) / (k as f64);
		}
		r0 += (x / (2 * n + 1) as f64) * r1;
	}
	(2. / PI.sqrt()) * r0
}
