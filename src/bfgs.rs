use leonhard::linear_algebra::*;

fn line_search<A>(x: Vector::<f64>, direction: Vector::<f64>, func: A, mut a: f64, p: f64,
	max_steps: usize) -> f64 where A: Fn(Vector::<f64>) -> f64 {
	let start_y = func(x.clone());
	for _ in 0..max_steps {
		if func(x.clone() + direction.clone() * a) - start_y >= a {
			break;
		}
		a *= p;
	}
	a
}

pub fn compute_new_inverse_hessian(curr_mat: Matrix::<f64>, s: Vector::<f64>, y: Vector::<f64>)
	-> Matrix::<f64> {
	let s_y = s.dot(&y);
	let y_s = y.outer_product(&s);
	let s_s = s.outer_product(&s);

	let mut y_mat = y.to_matrix();
	let y_transpose = y_mat.transpose();

	let a = (s_s * (s_y + y.dot(&(curr_mat.clone() * y.clone())))) / (s_y * s_y);
	let b = (curr_mat.clone() * y_s + s.to_matrix() * (y_transpose.clone() * curr_mat.clone()))
		/ s_y;

	curr_mat + a - b
}

pub fn bfgs<A, B>(start: Vector::<f64>, func: A, gradient: B, max_steps: usize) -> Vector::<f64>
	where
		A: Fn(Vector::<f64>) -> f64 + Copy,
		B: Fn(Vector::<f64>) -> Vector::<f64> + Copy {
	let mut x = start.clone();
	let mut inverse_hessian_matrix = Matrix::identity(start.get_size());

	for _ in 0..max_steps {
		let direction = inverse_hessian_matrix.clone() * gradient(x.clone());
		let alpha = line_search(x.clone(), direction.clone(), func, 10., 0.8, 1024);
		let s = direction * alpha;
		let x_next = x.clone() + s.clone();
		let gradient_difference = gradient(x_next) - gradient(x.clone());
		inverse_hessian_matrix = compute_new_inverse_hessian(inverse_hessian_matrix, s,
			gradient_difference);
	}

	Vector::new(0)
}
