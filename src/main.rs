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

fn rand_vector(size: usize, amplitude: f64) -> Vector<f64> {
	let mut vec = Vec::<f64>::new();
	for _ in 0..size {
		vec.push((pseudo_rand() - 0.5) * (amplitude * 2.));
	}
	Vector::<f64>::from_vec(vec)
}

fn bayesian_optimization<F: Fn(Vector<f64>) -> f64>(f: F, dim: usize, n_0: usize, n: usize)
	-> Vector<f64> {
	assert!(n_0 > 0);
	assert!(n_0 <= n);

	let mut data = Vec::<(Vector::<f64>, f64)>::new();

	for _ in 0..n_0 {
		let x = rand_vector(dim, 100.);
		data.push((x.clone(), f(x)));
	}
	for _ in n_0..n {
		// TODO Update the posterior probability distribution on f using all available data
		// TODO Let x[n] be a maximizer of the acquisition function over x, where the acquisition function is computed using the cuurrent posterior distribution
		let x = Vector::<f64>::from_vec(vec!{ 0. }); // TODO
		data.push((x.clone(), f(x)));
	}

	let mut max_index = 0;
	for i in 0..data.len() {
		println!("{} {}", data[i].0, data[i].1);
		if data[i].1 > data[max_index].1 {
			max_index = i;
		}
	}

	data[max_index].0.clone()
}

fn main() {
	let result = bayesian_optimization(| v | {
		let x = v.x();
		((x - 10.) * x).sin() - ((x - 10.) * (x - 10.)) + 10.
	}, 1, 10, 50);
	println!("Result: {}", result);
}
