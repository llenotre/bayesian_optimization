use leonhard::linear_algebra::*;

fn line_search<A>(x: Vector::<f64>, direction: Vector::<f64>, func: A, mut a: f64, c: f64, p: f64,
	max_steps: usize) -> f64 where A: Fn(Vector::<f64>) -> f64 {
	let start_y = func(x.clone());
	for _ in 0..max_steps {
		if start_y - func(x.clone() + direction.clone() * a) >= a * -c {
			break;
		}
		a *= p;
	}
	a
}

pub fn bfgs<A, B>(start: Vector::<f64>, func: A, gradient: B, max_steps: usize) -> Vector::<f64>
	where
		A: Fn(Vector::<f64>) -> f64 + Copy,
		B: Fn(Vector::<f64>) -> Vector::<f64> + Copy {
	let mut x = start.clone();
	let mut inverse_hessian_matrix = Matrix::identity(start.get_size());

	for _ in 0..max_steps {
		let direction = inverse_hessian_matrix.clone() * gradient(x.clone());
		let alpha = line_search(x.clone(), direction.clone(), func, 10., 0.5, 0.8, 1024);
		let s = direction * alpha;
		let x_next = x.clone() + s;
		let gradient_difference = gradient(x_next) - gradient(x.clone());
		// TODO Compute new inverse hessian matrix
	}

	Vector::new(0)
}
