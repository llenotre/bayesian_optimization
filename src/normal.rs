use std::f64::consts::PI;

use crate::util;

pub fn pdf(x: f64, mean: f64, std_deviation: f64) -> f64 {
	let a = (x - mean) / std_deviation;
	let b = std_deviation * (2. * PI).sqrt();
	(-0.5 * a * a).exp() / b
}

pub fn cdf(x: f64, mean: f64, std_deviation: f64) -> f64 {
	let integral_constant = -(PI / 2.).sqrt() * std_deviation;
	let F_x = integral_constant * util::erf((mean - x) / 2_f64.sqrt() * std_deviation);
	let F_minus_inf = integral_constant * 1.; // `1` is the limit of erf(x) for x -> +inf
	let b = std_deviation * (2. * PI).sqrt();
	(F_x - F_minus_inf) / b
}
