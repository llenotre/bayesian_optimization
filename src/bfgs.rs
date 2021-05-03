use leonhard::linear_algebra::*;

fn approximate_gradient<A>(func: A, x: Vector<f64>) -> Vector<f64>
	where A: Fn(Vector::<f64>) -> f64 + Copy {
	let dim = x.get_size();
	let step_size = 0.1; // TODO
	let mut v = Vector::<f64>::new(dim);

	for i in 0..dim {
		let mut direction = Vector::<f64>::new(dim);
		direction[i] = step_size;

		v[i] = func(x.clone() + direction.clone()) - func(x.clone());
		println!("diff: {} - {}", func(x.clone() + direction), func(x.clone()));
	}

	println!("Derivative at {}:\n{}", x, v);

	v
}

fn gradient_descent<A>(mut x: Vector::<f64>, gradient: A, step_size: f64, max_steps: usize)
	-> Vector::<f64> where A: Fn(Vector::<f64>) -> Vector::<f64> {
	for _ in 0..max_steps {
		//println!("-> {} {}", x.clone(), gradient(x.clone()));
		x -= gradient(x.clone()) * step_size;
	}
	x
}

fn line_search<A>(x: Vector::<f64>, direction: Vector::<f64>, func: A, control: f64, begin: f64,
	step_size: f64, max_steps: usize) -> f64 where A: Fn(Vector::<f64>) -> f64 {
	let local_curvature = (-direction.clone()).dot(&x);
	let t = -control * local_curvature;
	let mut j = 0;
	let mut a = begin;
	while j < max_steps && func(x.clone()) - func(x.clone() + direction.clone() * a) < a * t {
		a = step_size * a;
		j += 1;
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

pub fn bfgs<A>(start: Vector::<f64>, func: A, max_steps: usize) -> Vector::<f64>
	where A: Fn(Vector::<f64>) -> f64 + Copy {
	let mut x = start.clone();
	let mut inverse_hessian_matrix = Matrix::identity(start.get_size());

	let gradient = | x | {
		approximate_gradient(func, x)
	};

	for _ in 0..max_steps {
		println!("x: {}", x.clone());
		let grad = gradient(x.clone());
		let direction = inverse_hessian_matrix.clone() * -grad.clone();
		println!("inverse_hessian_matrix: {}", inverse_hessian_matrix.clone());
		println!("direction: {}", direction.clone());
		let alpha = line_search(x.clone(), direction.clone(), func, 1., 100., 0.1, 1024);
		println!("alpha: {}", alpha);
		let s = direction * alpha;
		let x_next = x.clone() + s.clone();
		let gradient_difference = gradient(x_next.clone()) - grad;
		//if gradient_difference.length() < 0.000001 {
			//break;
		//}
		x = x_next;
		//println!("gradient_difference: {}", gradient_difference);
		inverse_hessian_matrix = compute_new_inverse_hessian(inverse_hessian_matrix, s,
			gradient_difference);
		println!("new inverse_hessian_matrix: {}", inverse_hessian_matrix.clone());
		println!("------------------------------------------------------------------------");
	}

	x
}

/*
pub fn bfgs<A>(start: Vector::<f64>, func: A, max_steps: usize) -> Vector::<f64>
	where A: Fn(Vector::<f64>) -> f64 + Copy {

	let gradient = | x | {
		approximate_gradient(func, x)
	};

	let dim = start.get_size();
	let mut x = start;
	let mut pos_diff = Vector::<f64>::new(dim);
	let mut grad_diff = Vector::<f64>::new(dim);

	for _ in 0..max_steps {
		let grad = gradient(x.clone());
		let rho = 1. / grad_diff.dot(&pos_diff);

		let mut q = grad.clone();
		let mut alpha = 0.;
		for _ in 0..100 { // TODO
			alpha = rho * pos_diff.dot(&q);
			q -= grad_diff.clone() * alpha;
		}

		let gamma = pos_diff.dot(&grad_diff) / grad_diff.dot(&grad_diff);
		let mut inverse_hessian = Matrix::<f64>::new(dim, dim) * gamma;
		let mut z = inverse_hessian * q;

		for _ in 0..100 { // TODO
			let beta = rho * grad_diff.dot(&z);
			z += pos_diff.clone() * (alpha - beta);
		}

		let new_pos = x.clone() + z;
		let new_grad = gradient(new_pos.clone());
		pos_diff = new_pos.clone() - x;
		grad_diff = new_grad - grad.clone();

		x = new_pos;
	}

	x
}
*/
