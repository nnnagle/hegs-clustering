// Correlated Embedded Topic Model with Covariates (keywords only)
// Sparse Y input as triplets (i, j, v) -> converted to CSR in transformed data.
//
// Likelihood (multinomial, sparse evaluation):
//   log p(y_d | theta_d, Phi) = lgamma(n_d + 1)
//                               - sum_{(d,k)∈nz} lgamma(y_{dk} + 1)
//                               + sum_{(d,k)∈nz} y_{dk} * log( Phi[k] * theta_d )
//
// Topic correlations: low-rank + diagonal factor model for logistic-normal
//   eta_d = (X W + b) + Lambda z_d + sigma_eta .* eps_d
//
// Shapes (typical): D≈2000, K≈4000, L≈40, M=200 (or 768), P=200, R∈[5,10]

data {
  // Sizes
  int<lower=1> D;                 // documents
  int<lower=1> K;                 // keywords
  int<lower=1> L;                 // topics
  int<lower=1> M;                 // keyword embedding dim (e.g., 200)
  int<lower=1> P;                 // doc covariate dim (e.g., 200)
  int<lower=1> R;                 // low-rank factor dim (R << L)

  // Sparse triplet for counts Y (must be consolidated: no duplicate (doc,kw) pairs)
  int<lower=0> NNZ;               // number of nonzeros
  int<lower=1, upper=D> ii[NNZ];  // doc indices (1-based)
  int<lower=1, upper=K> jj[NNZ];  // keyword indices (1-based)
  int<lower=1>          vv[NNZ];  // counts

  // Keyword and document features
  matrix[K, M] E;                 // keyword embeddings (rows L2-normalized recommended)
  matrix[D, P] X;                 // doc PCA covariates (columns standardized recommended)

  // Prior scales
  real<lower=0> beta_scale;       // e.g., 0.5
  real<lower=0> W_scale;          // e.g., 0.1
  real<lower=0> Lambda_scale;     // e.g., 0.5
}

transformed data {
  // Convert triplets to CSR-by-document for O(NNZ) passes per epoch
  int row_ptr[D + 1];             // 1-based start pointers; row_ptr[D+1] = NNZ + 1
  int col_ind[NNZ];               // keyword indices ordered by doc
  int val[NNZ];                   // counts ordered by doc

  // Build row_ptr via counting sort on ii
  int counts[D];
  int next[D];

  for (int d = 1; d <= D; d++) counts[d] = 0;
  for (int n = 1; n <= NNZ; n++) counts[ii[n]] += 1;

  row_ptr[1] = 1;
  for (int d = 1; d <= D; d++)
    row_ptr[d + 1] = row_ptr[d] + counts[d];

  for (int d = 1; d <= D; d++) next[d] = 0;

  for (int n = 1; n <= NNZ; n++) {
    int d = ii[n];
    int idx = row_ptr[d] + next[d];
    col_ind[idx] = jj[n];
    val[idx] = vv[n];
    next[d] += 1;
  }

  // Per-document multinomial constants:
  // C_d = lgamma(n_d + 1) - sum_{nz} lgamma(y_{dk} + 1)
  int N_doc[D];
  vector[D] log_fact_const;
  for (int d = 1; d <= D; d++) {
    int s = row_ptr[d];
    int e = row_ptr[d + 1] - 1;
    int n_sum = 0;
    real c = 0;
    for (int t = s; t <= e; t++) {
      n_sum += val[t];
      c -= lgamma(val[t] + 1);
    }
    N_doc[d] = n_sum;
    log_fact_const[d] = c + lgamma(n_sum + 1);
  }
}

parameters {
  // Topic embeddings in keyword embedding space
  matrix[M, L] B_raw;
  real<lower=0> tau_beta;               // global scale

  // Keyword softmax temperature
  real<lower=0> kappa;

  // Covariate map for eta
  matrix[P, L] W;
  vector[L] b;

  // Low-rank + diagonal correlation structure
  matrix[L, R] Lambda;                  // factor loadings
  vector<lower=0>[L] sigma_eta;         // uniq std devs

  // Per-document latent factors (non-centered)
  matrix[D, R] z;                       // z_d ~ N(0, I_R)
  matrix[D, L] eps;                     // eps_d ~ N(0, I_L)
}

transformed parameters {
  // Topic embeddings and keyword logits
  matrix[M, L] B = tau_beta * B_raw;    // (M x L)
  matrix[K, L] A = E * B;               // (K x L)

  // Topic → keyword distributions (columns sum to 1)
  matrix[K, L] Phi;
  for (int l = 1; l <= L; l++) {
    Phi[, l] = softmax(kappa * A[, l]);
  }

  // Mean (by covariates) for eta
  matrix[D, L] mu = X * W + rep_matrix(b', D);
}

model {
  // ---- Priors ----
  to_vector(B_raw) ~ normal(0, 1);
  tau_beta ~ normal(0, beta_scale) T[0, ];

  kappa ~ lognormal(0, 0.5);

  to_vector(W) ~ normal(0, W_scale);
  b ~ normal(0, 1);

  to_vector(Lambda) ~ normal(0, Lambda_scale);
  sigma_eta ~ lognormal(0, 0.5);

  to_vector(z) ~ normal(0, 1);
  to_vector(eps) ~ normal(0, 1);

  // ---- Sparse multinomial likelihood ----
  for (int d = 1; d <= D; d++) {
    vector[L] eta_d = (mu[d]') + Lambda * (z[d]') + sigma_eta .* (eps[d]');
    vector[L] theta_d = softmax(eta_d);

    // Start with per-doc constant
    real lp = log_fact_const[d];

    // Add only nnz terms: y_{dk} * log(Phi[k] * theta_d)
    int s = row_ptr[d];
    int e = row_ptr[d + 1] - 1;
    for (int t = s; t <= e; t++) {
      int k = col_ind[t];
      real pk = Phi[k] * theta_d;       // row_vector[L] * vector[L] -> real
      lp += val[t] * log(pk);
    }
    target += lp;
  }
}

generated quantities {
  matrix[D, L] Theta;
  vector[D] log_lik;

  for (int d = 1; d <= D; d++) {
    vector[L] eta_d = (mu[d]') + Lambda * (z[d]') + sigma_eta .* (eps[d]');
    vector[L] theta_d = softmax(eta_d);
    Theta[d] = (to_row_vector(theta_d));

    // Sparse log-likelihood using same constants
    real lp = log_fact_const[d];
    int s = row_ptr[d];
    int e = row_ptr[d + 1] - 1;
    for (int t = s; t <= e; t++) {
      int k = col_ind[t];
      real pk = Phi[k] * theta_d;
      lp += val[t] * log(pk);
    }
    log_lik[d] = lp;
  }
}
