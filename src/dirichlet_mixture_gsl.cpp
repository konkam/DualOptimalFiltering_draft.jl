// [[Rcpp::depends(RcppGSL)]]
// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include <RcppGSL.h>
#include <gsl/gsl_randist.h>
using namespace Rcpp;


// // [[Rcpp::export]]
// double meanC(NumericVector x) {
//   int n = x.size();
//   double total = 0;
//
//   for(int i = 0; i < n; ++i) {
//     total += x[i];
//   }
//   return total / n;
// }

// // [[Rcpp::export]]
// double ddirichlet_gsl(const RcppGSL::vector<double> & x, const RcppGSL::vector<double> & alpha, int K) {
//   gsl_ran_dirichlet_pdf(K, alpha, x)
// }

// // [[Rcpp::export]]
// double ddirichlet_gsl(const NumericVector x, const NumericVector alpha) {
//   int K = x.size();
//   gsl_ran_dirichlet_pdf(K, alpha.begin(), x.begin());
// }

// [[Rcpp::export]]
double ddirichlet_gsl_arma(const arma::vec x, const arma::vec alpha) {
  int K = x.size();
  gsl_ran_dirichlet_pdf(K, alpha.begin(), x.begin());
}

// // [[Rcpp::export]]
// double ddirichlet_mixture_gsl(const NumericVector x, const NumericVector alpha, const NumericVector weights) {
//   double total = 0;
//   int n = weights.size();
//   int K = x.size();
//   NumericVector alpha_i(K);
//   for(int i = 0; i < n; ++i) {
//     for(int j=0; j < K; ++j){
//       // Rcout << "The value j " << j << std::endl;
//       // Rcout << "The value idx " << i*K+j << std::endl;
//       // Rcout << "The value alpha[j] " << alpha[j] << std::endl;
//       // Rcout << "The value lambda[i*K+j] " << lambda[i*K+j] << std::endl;
//       alpha_i[j] = alpha[i*K+j];
//     }
//     // auto alpha_i = gsl::as_span(alpha).subspan(i*K, (i+1)*K-1));
//     total += weights[i] * ddirichlet_gsl(x, alpha_i);
//   }
//   return total;
// }

// [[Rcpp::export]]
double ddirichlet_mixture_gsl_arma(const arma::vec x, const arma::vec alpha, const arma::vec weights) {
  double total = 0;
  int n = weights.size();
  int K = x.size();
  for(int i = 0; i < n; ++i) {
    // auto alpha_i = gsl::as_span(alpha).subspan(i*K, (i+1)*K-1));
    total += weights[i] * ddirichlet_gsl_arma(x, alpha.subvec( i*K,  (i+1)*K-1));
  }
  return total;
}

// // [[Rcpp::export]]
// double ddirichlet_mixture_gsl_alpha_m(const NumericVector x, const NumericVector alpha, const NumericVector weights, const NumericVector lambda) {
//   double total = 0;
//   int n = weights.size();
//   int K = x.size();
//   NumericVector alpha_plus_m(K);
//   for(int i = 0; i < n; ++i) {
//     // NumericVector alpha_plus_m = alpha + lambda[i];
//     // alpha_plus_m = clone(alpha);
//     for(int j=0; j < K; ++j){
//       // Rcout << "The value j " << j << std::endl;
//       // Rcout << "The value idx " << i*K+j << std::endl;
//       // Rcout << "The value alpha[j] " << alpha[j] << std::endl;
//       // Rcout << "The value lambda[i*K+j] " << lambda[i*K+j] << std::endl;
//       alpha_plus_m[j] = alpha[j] + lambda[i*K+j];
//     }
//     // Rcout << "The value lambda " << alpha_plus_m << std::endl;
//     // Rcout << "The value alpha_plus_m" << alpha_plus_m << std::endl;
//     total += weights[i] * ddirichlet_gsl(x, alpha_plus_m);
//   }
//   return total;
// }


// // [[Rcpp::export]]
// double ddirichlet_mixture_gsl_alpha_m_arma(const arma::vec x, const arma::vec alpha, const arma::vec weights, const arma::vec lambda) {
//   int n = weights.size();
//   int K = x.size();
//   arma::vec alpha_plus_m(lambda.size());
//   for(int i = 0; i < n; ++i) {
//     for(int j=0; j < K; ++j){
//       alpha_plus_m[i*K+j] = alpha[j] + lambda[i*K+j];
//     }
//   }
//   return ddirichlet_mixture_gsl_arma(x, alpha_plus_m, weights);
// }
