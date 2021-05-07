use leonhard::linear_algebra::*;

use bfgs::bfgs;

mod bfgs;
mod normal;
mod util;

struct Sample {
	x: Vector::<f64>,
	y: f64,
}

fn gaussian_kernel(a: Vector::<f64>, b: Vector::<f64>) -> f64 {
	let alpha = 1.; // TODO
	let dist = (a - b).length();
	alpha * (-(dist * dist)).exp()
}

fn build_covariance_matrix<F0: Fn(usize) -> Vector::<f64>, F1: Fn(usize) -> Vector::<f64>>(y: F0,
	height: usize, x: F1, width: usize) -> Matrix::<f64> {
	let mut m = Matrix::<f64>::new(height, width);

	for i in 0..height {
		for j in 0..width {
			let val = gaussian_kernel(y(i), x(j));
			*m.get_mut(i, j) = val;
			if j < height && i < width {
				*m.get_mut(j, i) = val;
			}
		}
	}

	m
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
	let a = build_covariance_matrix(| _ | {
		x.clone()
	}, 1, | i | {
		data[i].x.clone()
	}, data.len());

	let b = build_covariance_matrix(| i | {
		data[i].x.clone()
	}, data.len(), | i | {
		data[i].x.clone()
	}, data.len()).get_inverse();

	let c = data_y_to_vec(data);

	//println!("m: {} * ({} * {}) = {}", a.clone(), b.clone(), c.clone(), a.clone() * (b.clone() * c.clone()));
	a * (b * c)
}

fn compute_std_deviation(data: &Vec<Sample>, x: Vector::<f64>) -> f64 {
	let a = gaussian_kernel(x.clone(), x.clone());

	let b = build_covariance_matrix(| _ | {
		x.clone()
	}, 1, | i | {
		data[i].x.clone()
	}, data.len());

	let c = build_covariance_matrix(| i | {
		data[i].x.clone()
	}, data.len(), | i | {
		data[i].x.clone()
	}, data.len()).get_inverse();

	let d = build_covariance_matrix(| i | {
		data[i].x.clone()
	}, data.len(), | _ | {
		x.clone()
	}, 1);

	a - (b.clone() * (c.clone() * d.clone())).get(0, 0).sqrt()
}

fn expected_improvement(mean: Vector::<f64>, std_deviation: f64, max_sample: f64)
	-> Vector::<f64> {
	let dim = mean.get_size();
	let mut v = Vector::<f64>::new(dim);

	for i in 0..dim {
		if std_deviation.abs() > 0.0001 {
			let delta = mean.get(i) - max_sample;

			let A = delta / std_deviation;

			let density = normal::pdf(A, *mean.get(i), std_deviation);
			let cumulative_density = normal::cdf(A, *mean.get(i), std_deviation);

			*v.get_mut(i) = util::maxf(delta, 0.) + std_deviation * density - delta.abs() * cumulative_density;
			//println!("max({}, 0) + {} * {} - |{}| * {} = {}", delta, std_deviation, density, delta, cumulative_density, *v.get(i));
		}
	}
	v
}

fn bayesian_optimization<F: Fn(Vector<f64>) -> f64>(f: F, dim: usize, n_0: usize, n: usize)
	-> Vector<f64> {
	assert!(n_0 > 0);
	assert!(n_0 <= n);

	let mut data = Vec::<Sample>::new();
	let mut max_index = 0;

	for _ in 0..n_0 {
		let x = util::rand_vector(dim, 10.);
		data.push(Sample {
			x: x.clone(),
			y: f(x)
		});
		println!("{}, {}", data[data.len() - 1].x.x(), data[data.len() - 1].y);
		if data[data.len() - 1].y > data[max_index].y {
			max_index = data.len() - 1;
		}
	}

	for _ in n_0..n {
		let max_sample = data[max_index].y;

		// TODO rm
		for i in -100..100 {
			let x = Vector::<f64>::from_vec(vec!{ i as f64 * 0.1 });
			let mean = compute_mean(&data, x.clone());
			let std_deviation = compute_std_deviation(&data, x.clone());
			//println!("{}, {}", *x.x(), expected_improvement(mean, std_deviation, max_sample).length());
			//println!("{}, {}", *x.x(), gaussian_kernel(x.clone(), Vector::new(1)));
		}

		let start = Vector::<f64>::new(dim); // TODO Init at random value?
		let x = bfgs(start, | x | {
			let mean = compute_mean(&data, x.clone());
			let std_deviation = compute_std_deviation(&data, x);
			-expected_improvement(mean, std_deviation, max_sample).length()
		}, 100);
		//println!("next: {}", x);

		data.push(Sample {
			x: x.clone(),
			y: f(x)
		});
	}

	data[max_index].x.clone()
}

fn main() {
	let result = bayesian_optimization(| v | {
		let x = v.x();
		(x * x).sin() * x - x * x
	}, 1, 10, 25);

	/*let result = bfgs(Vector::<f64>::from_vec(vec!{20.}), | x | {
		x.x() * x.x() + x.x() * 2. + x.x().cos() * 80.
	}, | x | {
		x.clone() * 2. + 2. - 80. * x.x().sin()
	}, 1024);*/

	println!("Result: {}", result);
}
