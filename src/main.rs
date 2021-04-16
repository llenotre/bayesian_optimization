use std::f64::consts::PI;

use leonhard::linear_algebra::*;

use bfgs::bfgs;

mod bfgs;
mod util;

struct Sample {
	x: Vector::<f64>,
	y: f64,
}

fn gaussian_kernel(a: Vector::<f64>, b: Vector::<f64>) -> f64 {
	let alpha = 1.; // TODO
	alpha * (-(a - b).length_squared()).exp()
}

// TODO gaussian_kernel gradient

fn build_covariance_matrix(x: Vector::<f64>, y: Vector::<f64>) -> Matrix::<f64> {
	assert!(x.get_size() == y.get_size());
	let size = x.get_size();
	let mut m = Matrix::new(size, size);

	for i in 0..size {
		for j in i..size {
			let val = gaussian_kernel(From::from(*x.get(i)), From::from(*y.get(j)));
			*m.get_mut(i, j) = val;
			if i != j {
				*m.get_mut(j, i) = val;
			}
		}
	}
	m
}

fn normal_density(x: f64, mean: f64, std_deviation: f64) -> f64 {
	let a = (x - mean) / std_deviation;
	let b = std_deviation * (2. * PI).sqrt();
	(-0.5 * a * a).exp() / b
}

fn normal_cdf(x: f64, mean: f64, std_deviation: f64) -> f64 {
	let integral_constant = -(PI / 2.).sqrt() * std_deviation;
	let F_x = integral_constant * util::erf((mean - x) / 2_f64.sqrt() * std_deviation);
	let F_minus_inf = integral_constant * 1.; // `1` is the limit of erf(x) for x -> +inf
	let b = std_deviation * (2. * PI).sqrt();
	(F_x - F_minus_inf) / b
}

fn data_x_to_vec(data: &Vec<Sample>) -> Vector::<f64> {
	// TODO
	Vector::<f64>::new(0)
}

fn data_y_to_vec(data: &Vec<Sample>) -> Vector::<f64> {
	let len = data.len();
	let mut v = Vector::new(len);

	for i in 0..len {
		v[i] = data[i].y;
	}
	v
}

fn compute_mean(data: &Vec<Sample>, x: Vector::<f64>) -> Vector::<f64> {
	let data_x = data_x_to_vec(data);
	let data_y = data_y_to_vec(data);

	let a = build_covariance_matrix(x.clone(), data_x.clone());
	let b = build_covariance_matrix(data_x.clone(), data_x).get_inverse();
	let c = data_y;
	a * (b * c)
}

fn compute_std_deviation(data: &Vec<Sample>, x: Vector::<f64>) -> Vector::<f64> {
	let data_x = data_x_to_vec(data);

	let a = build_covariance_matrix(x.clone(), x.clone());
	let b = build_covariance_matrix(x.clone(), data_x.clone());
	let c = build_covariance_matrix(data_x.clone(), data_x.clone()).get_inverse();
	let d = build_covariance_matrix(data_x, x);
	(a - (b * (c * d))).to_vector() // TODO sqrt
}

fn expected_improvement(mean: Vector::<f64>, std_deviation: Vector::<f64>, max_sample: f64)
	-> Vector::<f64> {
	let dim = mean.get_size();
	let mut v = Vector::<f64>::new(dim);
	for i in 0..dim {
		let delta = mean.get(i) - max_sample;
		let density = normal_density(delta / std_deviation.get(i), *mean.get(i), *std_deviation.get(i));
		let cumulative_density = normal_cdf(delta / std_deviation.get(i), *mean.get(i), *std_deviation.get(i));
		*v.get_mut(i) = util::maxf(delta, 0.) + std_deviation.get(i) * density - delta.abs() * cumulative_density
	}
	v
}

fn expected_improvement_gradient(mean: Vector::<f64>, std_deviation: Vector::<f64>,
	max_sample: f64) -> Vector::<f64> {
	let dim = mean.get_size();
	// TODO
	Vector::new(dim)
}

fn bayesian_optimization<F: Fn(Vector<f64>) -> f64>(f: F, dim: usize, n_0: usize, n: usize)
	-> Vector<f64> {
	assert!(n_0 > 0);
	assert!(n_0 <= n);

	let mut data = Vec::<Sample>::new();

	for _ in 0..n_0 {
		let x = util::rand_vector(dim, 100.);
		data.push(Sample {
			x: x.clone(),
			y: f(x)
		});
	}
	for _ in n_0..n {
		let max_sample = 0.; // TODO
		let start = Vector::<f64>::new(dim); // TODO

		let x = bfgs(start, | x | {
			let mean = compute_mean(&data, x.clone());
			let std_deviation = compute_std_deviation(&data, x);
			expected_improvement(mean, std_deviation, max_sample).length()
		}, | x | {
			let mean = compute_mean(&data, x.clone());
			let std_deviation = compute_std_deviation(&data, x);
			expected_improvement_gradient(mean, std_deviation, max_sample)
		}, 1024);

		data.push(Sample {
			x: x.clone(),
			y: f(x)
		});
	}

	let mut max_index = 0;
	for i in 0..data.len() {
		println!("{} {}", data[i].x, data[i].y);
		if data[i].y > data[max_index].y {
			max_index = i;
		}
	}

	data[max_index].x.clone()
}

fn main() {
	let result = bayesian_optimization(| v | {
		let x = v.x();
		((x - 10.) * x).sin() - ((x - 10.) * (x - 10.)) + 10.
	}, 1, 10, 50);

	/*let result = bfgs(Vector::<f64>::from_vec(vec!{10.}), | x | {
		x.x() * x.x() + x.x() * 2. + x.x().cos() * 80.
	}, | x | {
		x.clone() * 2. + 2. - 80. * x.x().sin()
	}, 1024);*/

	println!("Result: {}", result);
}
