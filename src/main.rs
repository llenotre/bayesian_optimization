use leonhard::linear_algebra::*;

use std::f64::consts::PI;

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

fn maxf(a: f64, b:  f64) -> f64 {
	if a > b {
		a
	} else {
		b
	}
}

fn erf(x: f64) -> f64 {
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

fn gaussian_kernel(a: Vector::<f64>, b: Vector::<f64>) -> f64 {
	let alpha = 1.; // TODO
	alpha * (-(a - b).length_squared()).exp()
}

/*fn build_covariance_matrix(x: Vector::<f64>, values: Matrix::<f64>) -> Matrix::<f64> {
	// TODO
}*/

fn build_covariance_matrix_symetric(values: Vector::<f64>) -> Matrix::<f64> {
	let size = values.get_size();
	let mut m = Matrix::new(size, size);

	for i in 0..size {
		for j in i..size {
			let val = gaussian_kernel(From::from(*values.get(i)), From::from(*values.get(j)));
			*m.get_mut(i, j) = val;
			if i != j {
				*m.get_mut(j, i) = val;
			}
		}
	}
	m
}

/*fn compute_mean() -> Vector::<f64> {
	// TODO
}

fn compute_std_deviation() -> Vector::<f64> {
	// TODO
}*/

fn normal_density(x: f64, mean: f64, std_deviation: f64) -> f64 {
	let a = (x - mean) / std_deviation;
	let b = std_deviation * (2. * PI).sqrt();
	(-0.5 * a * a).exp() / b
}

fn normal_cdf(x: f64, mean: f64, std_deviation: f64) -> f64 {
	let integral_constant = -(PI / 2.).sqrt() * std_deviation;
	let F_x = integral_constant * erf((mean - x) / 2_f64.sqrt() * std_deviation);
	let F_minus_inf = integral_constant * 1.; // `1` is the limit of erf(x) for x -> +inf
	let b = std_deviation * (2. * PI).sqrt();
	(F_x - F_minus_inf) / b
}

fn expected_improvement(mean: Vector::<f64>, std_deviation: Vector::<f64>,
	max_sample: Vector::<f64>) -> Vector::<f64> {
	let mut v = Vector::<f64>::new(mean.get_size());

	for i in 0..v.get_size() {
		let delta = mean.get(i) - max_sample.get(i);
		let density = normal_density(delta / std_deviation.get(i), *mean.get(i), *std_deviation.get(i));
		let cumulative_density = normal_cdf(delta / std_deviation.get(i), *mean.get(i), *std_deviation.get(i));
		*v.get_mut(i) = maxf(delta, 0.) + std_deviation.get(i) * density - delta.abs() * cumulative_density
	}

	v
}

fn get_next_eval_point(mean: Vector::<f64>, _std_deviation: Vector::<f64>,
	_max_sample: Vector::<f64>) -> Vector::<f64> {
	// TODO

	Vector::<f64>::new(mean.get_size())
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
		// TODO Let x[n] be a maximizer of the acquisition function over x, where the acquisition function is computed using the current posterior distribution
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
