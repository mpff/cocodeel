# UKBB application figure — Panel A (resampling) + Panel B (AUC).
#
# Run from project root:
#   Rscript experiments/ukbb/figures/UKBB_application_fig.R

library(ggplot2)
library(dplyr)
library(tidyr)
library(mltools)
library(grid)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# Sample-split rexports and the K=2 cross-fit: refit_from_checkpoints.py + aggregate.py;
# the no-sample-split refit and the other methods: competitors_from_checkpoints.py and
# train_joint.py; the K=3 cross-fit: run_crossfit.py + aggregate.py.
DATA_DIR <- "experiments/ukbb/runs/refit/rexports/"
RUN_DIR  <- "experiments/ukbb/runs/refit/"
K3_DIR   <- "experiments/ukbb/runs/crossfit_k3/"
OUT_DIR  <- paste0(DATA_DIR, "../graphics/")
if (!dir.exists(OUT_DIR)) dir.create(OUT_DIR, recursive = TRUE)
OUT_PATH_MAIN       <- paste0(OUT_DIR, "Fig_UKBB_application_main.pdf")
OUT_PATH_APPENDIX   <- paste0(OUT_DIR, "Fig_UKBB_appendix.pdf")

# ---------------------------------------------------------------------------
# Colours — Okabe–Ito
# ---------------------------------------------------------------------------
OI_BLUE       <- "#0072B2"
OI_ORANGE     <- "#E69F00"
OI_VERMILLION <- "#D55E00"   # base / No Control primary
OI_GREEN      <- "#009E73"   # refit_age / Age Control primary
OI_SKY        <- "#56B4E9"   # crossfit_age / Cross-fit Age Control primary
GREY_REF      <- "grey30"
# Pale (desaturated) variants — used for sample-split methods, where the
# saturated hue marks the cross-fit version of the same control.
SKY_PALE      <- "#A8DBF0"   # paler than OI_SKY — used for sample-split (single side)
GREY_NOSPLIT  <- "#BFBFBF"   # neutral grey for "no sample split" baseline (different from primary hues)
OI_BLUE_DARK  <- "#005580"   # darker than OI_SKY — used for K=3 cross-fit (distinct from K=2's OI_SKY)

# ---------------------------------------------------------------------------
# Shared theme (inherited from UKBB_plots.R)
# ---------------------------------------------------------------------------
shared_theme <- theme_bw() +
  theme(
    text              = element_text(size = 8, family = "serif"),
    axis.text.x       = element_text(size = 6),
    axis.text.y       = element_text(size = 6),
    strip.text        = element_text(size = 7, family = "serif"),
    legend.background = element_rect(color = NA, fill = NA),
    legend.text       = element_text(size = 7, family = "serif"),
    legend.title      = element_text(size = 7, family = "serif"),
    plot.margin       = margin(2, 2, 2, 2),
    plot.title        = element_text(size = 8, family = "serif",
                                     hjust = 0.5, face = "bold")
  )

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
actual <- read.csv(paste0(DATA_DIR, "testset.csv"))
folds  <- read.csv(paste0(DATA_DIR, "trainset_folds.csv"))

tp  <- read.csv(paste0(DATA_DIR, "testset_predictions.csv"))
tpc <- read.csv(paste0(DATA_DIR, "testset_predictions_controlled.csv"))

y_true_lookup <- actual %>% select(obs_id, y_true = y)

tp_reg <- tp %>%
  left_join(y_true_lookup, by = "obs_id") %>%
  mutate(marg = "regular") %>%
  select(method, coef, fold, y, y_true, marg)

tp_ctrl <- tpc %>%
  left_join(y_true_lookup, by = "obs_id") %>%
  rename(y = y_controlled) %>%
  mutate(marg = "controlled") %>%
  select(method, coef, fold, y, y_true, marg)

auc_df <- bind_rows(tp_reg, tp_ctrl) %>%
  group_by(method, coef, fold, marg) %>%
  summarize(auc = auc_roc(y, y_true), .groups = "drop") %>%
  mutate(training = factor(
    if_else(coef == 0, "Balanced training", "Confounded training"),
    levels = c("Balanced training", "Confounded training")
  ))

# Cross-fit results — kept as a separate data frame. crossfit_results.csv stores
# per-fold summary AUCs directly (auc = regular, auc_marg = controlled), with
# logit-space marginalization. We use this for panels comparing sample-split vs
# cross-fit, since both numerators are computed under the same marg definition.
# per-fold AUCs, regular and age-controlled (marginalised), for one cross-fit results file
read_crossfit <- function(path) {
  raw <- read.csv(path)
  bind_rows(
    raw %>% select(method, coef, fold, auc) %>% mutate(marg = "regular"),
    raw %>% select(method, coef, fold, auc = auc_marg) %>% mutate(marg = "controlled")
  ) %>%
    mutate(training = factor(
      if_else(coef == 0, "Balanced training", "Confounded training"),
      levels = c("Balanced training", "Confounded training")
    ))
}
crossfit_df <- read_crossfit(paste0(RUN_DIR, "crossfit_results.csv"))
# K=3: same schema, method names crossfit_k3_age / refit_split_age / refit_nosamp_age
k3_df <- read_crossfit(paste0(K3_DIR, "k3_crossfit_results.csv"))

# No-sample-split baseline: base_full's last layer refit with age on its own
# training sample (nosplit_age), AUC marginalised over the training age distribution.
no_split_df <- read.csv(paste0(RUN_DIR, "competitor_results.csv")) %>%
  filter(method == "nosplit_age") %>%
  transmute(coef, fold, auc = auc_marg,
            training = factor(
              if_else(coef == 0, "Balanced training", "Confounded training"),
              levels = c("Balanced training", "Confounded training")))

# ---------------------------------------------------------------------------
# Panel A — three density mini-panels
# ---------------------------------------------------------------------------
# x-range shared across all three densities
xlim_age <- range(c(actual$age, folds$age))

DIST_HIGH <- OI_ORANGE   # High-alc class (positive)
DIST_CTRL <- OI_BLUE     # Control class — kept distinct from sky-blue method colour by hue

make_density <- function(df, title, show_x_axis = FALSE) {
  p <- ggplot(df, aes(x = age, fill = factor(y))) +
    geom_density(alpha = 0.55, colour = NA, adjust = 1.2) +
    scale_fill_manual(values = c("0" = DIST_CTRL, "1" = DIST_HIGH), guide = "none") +
    coord_cartesian(xlim = xlim_age, expand = FALSE) +
    labs(title = title, x = NULL, y = NULL) +
    shared_theme +
    theme(
      axis.text.y       = element_blank(),
      axis.ticks.y      = element_blank(),
      panel.grid.minor  = element_blank(),
      panel.grid.major.y = element_blank(),
      plot.margin       = margin(1, 3, 1, 3)
    )
  if (!show_x_axis) {
    p <- p + theme(
      axis.text.x  = element_blank(),
      axis.ticks.x = element_blank()
    )
  } else {
    p <- p + labs(x = "Age (years)") +
      theme(axis.title.x = element_text(margin = margin(t = 1)))
  }
  p
}

g_orig <- ggplotGrob(make_density(actual,                 "Original",    show_x_axis = TRUE))
g_bal  <- ggplotGrob(make_density(folds %>% filter(coef == 0), "Balanced",    show_x_axis = TRUE))
g_conf <- ggplotGrob(make_density(folds %>% filter(coef == 2), "Confounded",  show_x_axis = TRUE))

# Parent 0–1 canvas: stacked layout for single-column main figure.
# Top row: Original (left) + legend (right). Bottom row: Balanced + Confounded.
# Two curved down-arrows from Original mark the resampling step.
panel_A <- ggplot() +
  coord_cartesian(xlim = c(0, 1), ylim = c(0, 1), expand = FALSE, clip = "off") +
  # Top tile: Original (left half of canvas) — taller for readability
  annotation_custom(g_orig, xmin = 0.04, xmax = 0.46, ymin = 0.55, ymax = 1.00) +
  # Bottom tiles: Balanced + Confounded — taller for readability
  annotation_custom(g_bal,  xmin = 0.04, xmax = 0.46, ymin = 0.00, ymax = 0.45) +
  annotation_custom(g_conf, xmin = 0.54, xmax = 0.96, ymin = 0.00, ymax = 0.45) +
  # β=0 arrow: Orig → Balanced (short, straight down)
  annotate("curve",
           x = 0.20, xend = 0.20, y = 0.55, yend = 0.46,
           curvature = 0.001,
           arrow = arrow(length = unit(0.05, "in"), type = "closed"),
           colour = GREY_REF, linewidth = 0.35) +
  annotate("text", x = 0.13, y = 0.50,
           label = 'beta["age"] == 0', parse = TRUE,
           size = 2.2, family = "serif", colour = GREY_REF) +
  # β=-2 arrow: Orig → Confounded (short down-right arc; label sits next to it)
  annotate("curve",
           x = 0.43, xend = 0.55, y = 0.55, yend = 0.46,
           curvature = 0.35,
           arrow = arrow(length = unit(0.05, "in"), type = "closed"),
           colour = GREY_REF, linewidth = 0.35) +
  annotate("text", x = 0.62, y = 0.51,
           label = 'beta["age"] == -2.0', parse = TRUE,
           size = 2.2, family = "serif", colour = GREY_REF) +
  # Mini-legend (centred in the empty top-right quadrant)
  annotate("rect", xmin = 0.72, xmax = 0.76, ymin = 0.83, ymax = 0.93,
           fill = DIST_HIGH, alpha = 0.55, colour = NA) +
  annotate("text", x = 0.78, y = 0.88, hjust = 0, vjust = 0.5,
           label = "High alc.", size = 2.1, family = "serif", colour = GREY_REF) +
  annotate("rect", xmin = 0.72, xmax = 0.76, ymin = 0.67, ymax = 0.77,
           fill = DIST_CTRL, alpha = 0.55, colour = NA) +
  annotate("text", x = 0.78, y = 0.72, hjust = 0, vjust = 0.5,
           label = "Control", size = 2.1, family = "serif", colour = GREY_REF) +
  theme_void() +
  theme(plot.margin = margin(0, 2, 0, 2))

# ---------------------------------------------------------------------------
# Panel B — two AUC sub-panels
# ---------------------------------------------------------------------------
ref_bal <- auc_df %>%
  filter(method == "base_full", coef == 0, marg == "regular") %>%
  pull(auc) %>%
  median()

y_range <- c(0.48, 0.82)

make_auc_panel <- function(df, primary_label, ref_label, title, hue,
                           show_y = TRUE, show_x = TRUE, annotate_ref = FALSE) {
  p <- ggplot(df, aes(x = training, y = auc,
                      fill = variant, alpha = variant)) +
    geom_hline(yintercept = ref_bal, linetype = "dashed",
               linewidth = 0.4, colour = GREY_REF) +
    geom_boxplot(aes(group = interaction(training, variant)),
                 position = position_dodge(0.7),
                 outlier.shape = NA, width = 0.55, linewidth = 0.35,
                 colour = GREY_REF) +
    geom_point(aes(group = interaction(training, variant)),
               position = position_jitterdodge(
                 dodge.width = 0.7, jitter.width = 0.1, seed = 1),
               size = 0.8, shape = 21, colour = GREY_REF, stroke = 0.2) +
    scale_fill_manual(values = c("primary" = hue, "reference" = hue),
                      breaks = c("primary", "reference"),
                      labels = c(primary_label, ref_label),
                      name = NULL) +
    scale_alpha_manual(values = c("primary" = 0.85, "reference" = 0.22),
                       breaks = c("primary", "reference"),
                       labels = c(primary_label, ref_label),
                       name = NULL) +
    guides(
      fill  = guide_legend(override.aes = list(alpha = c(0.85, 0.22))),
      alpha = "none"
    ) +
    scale_y_continuous(breaks = seq(0.5, 0.8, 0.1)) +
    coord_cartesian(ylim = y_range) +
    labs(title = title, x = NULL,
         y = if (show_y) "AUC on balanced test data" else NULL) +
    shared_theme +
    theme(
      panel.grid.minor = element_blank(),
      legend.position  = c(0.02, 0.98),
      legend.justification = c(0, 1),
      legend.key.size  = unit(0.09, "in"),
      legend.margin    = margin(0, 2, 0, 2),
      legend.background = element_rect(fill = alpha("white", 0.7), colour = NA)
    )
  if (!show_y) {
    p <- p + theme(axis.text.y = element_blank(),
                   axis.ticks.y = element_blank())
  }
  if (!show_x) {
    p <- p + theme(axis.text.x = element_blank(),
                   axis.ticks.x = element_blank())
  }
  if (annotate_ref) {
    p <- p + annotate("text",
                      x = Inf, y = ref_bal,
                      hjust = 1.02, vjust = -0.4,
                      label = "Unconfounded target",
                      size = 2.5, family = "serif", colour = GREY_REF)
  }
  p
}

# =========================================================================
# Fig S2 data: coefficient recovery + lambda-path diagnostics
# =========================================================================
# True coefficients in the unstandardised / per-z_coef convention used in the
# DGP (logit slope on (age - mu_age)/std_age).  The CSV stores raw per-year
# coefficients (≈ -0.298); we multiply by STD_AGE to express them on the same
# scale as the true z_coef = -2.
TRUE_B_AGE <- -2.0
STD_AGE    <-  2.0 / 0.298   # implied std factor, ≈ 6.71

# All-estimator coefficients (sample-split, K=2, K=3) for the appendix coef
# recovery panels. No-sample-split coefficients are unavailable (the OLD
# 2026-03-30 run did not export fitted coefs, only predictions); extending to
# 4 estimators would require a refit there.
build_estimator_coefs <- function(df, method_name, estimator_label, b_age_col) {
  df %>%
    filter(method == method_name) %>%
    transmute(
      estimator = estimator_label,
      coef, fold,
      training = factor(
        if_else(coef == 0, "Balanced training", "Confounded training"),
        levels = c("Balanced training", "Confounded training")
      ),
      b_age = .data[[b_age_col]]
    )
}

all_estimators_coefs_df <- bind_rows(
  # Sample split — fitted_coefs.csv stores the age coefficient as `age`.
  build_estimator_coefs(
    read.csv(paste0(DATA_DIR, "fitted_coefs.csv")),
    "refit_age", "Sample split", b_age_col = "age"
  ),
  # Cross-fit K=2 — crossfit_results.csv has b_age.
  if (exists("cf_raw")) build_estimator_coefs(
    cf_raw, "crossfit_age", "Cross-fit (K=2)", b_age_col = "b_age"
  ) else data.frame(),
  # Cross-fit K=3 — k3_crossfit_results.csv same schema.
  if (exists("k3_raw")) build_estimator_coefs(
    k3_raw, "crossfit_k3_age", "Cross-fit (K=3)", b_age_col = "b_age"
  ) else data.frame()
) %>%
  mutate(estimator = factor(
    estimator,
    levels = c("Sample split", "Cross-fit (K=2)", "Cross-fit (K=3)")
  ))

# Age panel displays -β̂_age (sign-flipped for positive-up readability).
# True -β_age = +0.298 in Confounded training, 0 in Balanced.
true_age_df <- data.frame(
  training = factor(c("Balanced training", "Confounded training"),
                    levels = c("Balanced training", "Confounded training")),
  true_val = c(0, -TRUE_B_AGE)
)

lambda_paths <- read.csv(paste0(DATA_DIR, "refit_lambda_paths.csv")) %>%
  filter(method == "refit_age") %>%
  mutate(
    training = factor(if_else(coef == 0, "Balanced", "Confounded"),
                      levels = c("Balanced", "Confounded")),
    fold = factor(fold),
    converged = as.logical(converged)
  )

selected_pts <- lambda_paths %>%
  group_by(method, coef, fold) %>%
  filter(lambda == selected_lambda[1]) %>%
  slice(1) %>%
  ungroup()


# =========================================================================
# Fig S2 builders
# =========================================================================
# Coefficient recovery panel for the Age Control model, with three estimators
# dodged using the same palette as the AUC panels. No-sample-split coefs are not available
# (see comment near `all_estimators_coefs_df`).
coef_estimator_hues <- c(
  "Sample split"    = SKY_PALE,
  "Cross-fit (K=2)" = OI_SKY,
  "Cross-fit (K=3)" = OI_BLUE_DARK
)

make_coef_panel_estimators <- function(df, y_col, ref_df, title, y_label,
                                       show_legend, single_ref = FALSE) {
  p <- ggplot(df, aes(x = training, y = !!rlang::sym(y_col),
                      fill = estimator)) +
    geom_hline(yintercept = 0, linewidth = 0.3, colour = "grey75")
  if (single_ref) {
    p <- p + geom_hline(
      yintercept = ref_df$true_val[1],
      linetype = "dashed", colour = GREY_REF, linewidth = 0.5
    )
  } else {
    p <- p + geom_errorbar(
      data = ref_df, inherit.aes = FALSE,
      aes(x = training, ymin = true_val, ymax = true_val),
      linetype = "dashed", colour = GREY_REF,
      linewidth = 0.5, width = 0.8
    )
  }
  p <- p +
    geom_boxplot(position = position_dodge(0.7),
                 alpha = 0.85, width = 0.5,
                 linewidth = 0.35, outlier.shape = NA,
                 colour = GREY_REF) +
    geom_point(position = position_jitterdodge(
                 dodge.width = 0.7, jitter.width = 0.1, seed = 1),
               size = 0.7, shape = 21, colour = GREY_REF, stroke = 0.2) +
    scale_fill_manual(values = coef_estimator_hues, name = NULL) +
    labs(title = title, x = NULL, y = y_label) +
    shared_theme +
    theme(panel.grid.minor = element_blank())
  if (show_legend) {
    p <- p + theme(
      legend.position = c(0.02, 0.98),
      legend.justification = c(0, 1),
      legend.key.size = unit(0.09, "in"),
      legend.text     = element_text(size = 6, family = "serif"),
      legend.background = element_rect(fill = alpha("white", 0.7), colour = NA)
    )
  } else {
    p <- p + theme(legend.position = "none")
  }
  p
}

pS2_age <- make_coef_panel_estimators(
  all_estimators_coefs_df %>%
    filter(!is.na(b_age)) %>%
    mutate(b_age_neg = -b_age * STD_AGE),
  "b_age_neg", true_age_df,
  title   = "Age coefficient recovery",
  y_label = expression(-hat(beta)[age]),
  show_legend = TRUE
)
# Annotate true value on the reference line.
pS2_age <- pS2_age +
  annotate("text", x = 2, y = -TRUE_B_AGE, hjust = 1.15, vjust = -0.4,
           label = '"true "*-beta["age"]', parse = TRUE,
           size = 2.3, family = "serif", colour = GREY_REF)

pS2_lambda <- ggplot(lambda_paths,
                     aes(x = lambda, y = val_loss,
                         colour = fold, group = fold)) +
  geom_line(alpha = 0.9, linewidth = 0.7) +
  geom_vline(data = selected_pts,
             aes(xintercept = selected_lambda, colour = fold),
             linetype = "solid", alpha = 0.85, linewidth = 0.6) +
  scale_x_log10(name = expression(lambda ~ "(log scale)"),
                breaks = 10^seq(0, 6, 2)) +
  scale_y_log10(name = "Validation loss (log scale)") +
  coord_cartesian(xlim = c(1e1, 1e6)) +
  scale_color_viridis_d(option = "viridis", begin = 0.05, end = 0.85,
                        name = "Fold") +
  facet_grid(. ~ training, scales = "free_y") +
  shared_theme +
  theme(panel.grid.minor = element_blank(),
        legend.position = "right",
        strip.background = element_rect(fill = "grey95", colour = NA),
        plot.title = element_text(size = 8, family = "serif",
                                  hjust = 0, face = "bold")) +
  labs(title = "Validation loss along the λ path")


make_auc_panel_simple <- function(df, title, hue,
                                  show_y = TRUE, show_x = TRUE,
                                  annotate_ref = FALSE, compact = FALSE) {
  y_label <- if (show_y) {
    if (compact) "AUC (balanced test)" else "AUC on balanced test data"
  } else NULL
  p <- ggplot(df, aes(x = training, y = auc)) +
    geom_hline(yintercept = ref_bal, linetype = "dashed",
               linewidth = 0.4, colour = GREY_REF) +
    geom_boxplot(aes(group = training),
                 fill = hue, alpha = 0.85,
                 width = 0.45, linewidth = 0.35, outlier.shape = NA,
                 colour = GREY_REF) +
    geom_point(aes(group = training),
               position = position_jitter(width = 0.08, seed = 1),
               size = 0.8, shape = 21, fill = hue, colour = GREY_REF,
               alpha = 0.85, stroke = 0.2) +
    scale_y_continuous(breaks = seq(0.5, 0.8, 0.1)) +
    coord_cartesian(ylim = y_range) +
    labs(title = title, x = NULL, y = y_label) +
    shared_theme +
    theme(panel.grid.minor = element_blank())
  if (!show_y) {
    p <- p + theme(axis.text.y = element_blank(),
                   axis.ticks.y = element_blank())
  }
  if (!show_x) {
    p <- p + theme(axis.text.x = element_blank(),
                   axis.ticks.x = element_blank())
  } else if (compact) {
    p <- p + theme(axis.text.x = element_text(angle = 25, hjust = 1))
  }
  if (annotate_ref) {
    p <- p + annotate("text",
                      x = Inf, y = ref_bal,
                      hjust = 1.02, vjust = -0.4,
                      label = "Unconfounded target",
                      size = 2.2, family = "serif", colour = GREY_REF)
  }
  p
}

# =========================================================================
# Fig S3 data: per-fold correlation of fx with age
# =========================================================================
fx_corr_df <- tp %>%
  left_join(actual %>% select(obs_id, age), by = "obs_id") %>%
  group_by(method, coef, fold) %>%
  summarize(
    cor_age = cor(fx, age),
    .groups = "drop"
  ) %>%
  filter(method %in% c("base_full", "refit_age")) %>%
  mutate(
    method_label = factor(
      method,
      levels = c("base_full", "refit_age"),
      labels = c("No Control", "Age Control")
    ),
    training = factor(
      if_else(coef == 0, "Balanced training", "Confounded training"),
      levels = c("Balanced training", "Confounded training")
    )
  )

corr_hues <- c(
  "No Control"  = OI_VERMILLION,
  "Age Control" = OI_GREEN
)

make_corr_panel <- function(df, y_col, title, y_label, show_legend) {
  p <- ggplot(df, aes(x = training, y = !!rlang::sym(y_col),
                      fill = method_label)) +
    geom_hline(yintercept = 0, linewidth = 0.4, linetype = "dashed",
               colour = GREY_REF) +
    geom_boxplot(position = position_dodge(0.7),
                 alpha = 0.85, width = 0.55,
                 linewidth = 0.35, outlier.shape = NA,
                 colour = GREY_REF) +
    geom_point(position = position_jitterdodge(
                 dodge.width = 0.7, jitter.width = 0.1, seed = 1),
               size = 0.8, shape = 21, colour = GREY_REF, stroke = 0.2) +
    scale_fill_manual(values = corr_hues, name = NULL) +
    labs(title = title, x = NULL, y = y_label) +
    shared_theme +
    theme(panel.grid.minor = element_blank())
  if (show_legend) {
    p <- p + theme(
      legend.position = c(0.02, 0.98),
      legend.justification = c(0, 1),
      legend.key.size = unit(0.09, "in"),
      legend.background = element_rect(fill = alpha("white", 0.7), colour = NA)
    )
  } else {
    p <- p + theme(legend.position = "none")
  }
  p
}

pS3_age <- make_corr_panel(
  fx_corr_df, "cor_age",
  title   = "Correlation with age",
  y_label = expression(Corr(hat(f)[X] * ", " * age) ~ "on balanced test data"),
  show_legend = TRUE
)

no_ctrl <- auc_df %>%
  filter(method %in% c("base_full", "base_half"), marg == "regular") %>%
  mutate(variant = if_else(method == "base_full", "primary", "reference"))

# Sample-split vs cross-fit comparison data frames (marg AUC only).
# Keep a `variant` column for the dodge logic in make_auc_panel.
age_compare <- crossfit_df %>%
  filter(method %in% c("refit_age", "crossfit_age"), marg == "controlled") %>%
  mutate(variant = if_else(method == "crossfit_age", "primary", "reference"))

# Main figure: just cross-fit age control marg (single panel, two boxes).
crossfit_main <- crossfit_df %>%
  filter(method == "crossfit_age", marg == "controlled")

# --- Appendix Fig S1 (extended AUC: three sub-panels) ---
# Panels 2 and 3 compare sample-split vs cross-fit (two distinct hues per panel),
# so we use a dual-method panel function instead of the primary/reference variant
# scheme used by make_auc_panel.
make_auc_panel_dual <- function(df, primary_label, reference_label,
                                title, primary_hue, reference_hue,
                                show_y = TRUE, show_x = TRUE,
                                annotate_ref = FALSE) {
  hues <- c("primary" = primary_hue, "reference" = reference_hue)
  p <- ggplot(df, aes(x = training, y = auc,
                      fill = variant)) +
    geom_hline(yintercept = ref_bal, linetype = "dashed",
               linewidth = 0.4, colour = GREY_REF) +
    geom_boxplot(aes(group = interaction(training, variant)),
                 position = position_dodge(0.7),
                 outlier.shape = NA, width = 0.55, linewidth = 0.35,
                 alpha = 0.85, colour = GREY_REF) +
    geom_point(aes(group = interaction(training, variant)),
               position = position_jitterdodge(
                 dodge.width = 0.7, jitter.width = 0.1, seed = 1),
               size = 0.8, shape = 21, colour = GREY_REF, stroke = 0.2) +
    scale_fill_manual(values = hues,
                      breaks = c("primary", "reference"),
                      labels = c(primary_label, reference_label),
                      name = NULL) +
    scale_y_continuous(breaks = seq(0.5, 0.8, 0.1)) +
    coord_cartesian(ylim = y_range) +
    labs(title = title, x = NULL,
         y = if (show_y) "AUC on balanced test data" else NULL) +
    shared_theme +
    theme(
      panel.grid.minor = element_blank(),
      legend.position  = c(0.02, 0.98),
      legend.justification = c(0, 1),
      legend.key.size  = unit(0.09, "in"),
      legend.margin    = margin(0, 2, 0, 2),
      legend.background = element_rect(fill = alpha("white", 0.7), colour = NA)
    )
  if (!show_y) p <- p + theme(axis.text.y = element_blank(),
                              axis.ticks.y = element_blank())
  if (!show_x) p <- p + theme(axis.text.x = element_blank(),
                              axis.ticks.x = element_blank())
  if (annotate_ref) {
    p <- p + annotate("text",
                      x = Inf, y = ref_bal,
                      hjust = 1.02, vjust = -0.4,
                      label = "Unconfounded target",
                      size = 2.5, family = "serif", colour = GREY_REF)
  }
  p
}

pS1_nc <- make_auc_panel(no_ctrl,
                          "Training on full data set",
                          "Training on half data set",
                          "No Control",  hue = OI_VERMILLION,
                          show_y = TRUE, show_x = TRUE,
                          annotate_ref = FALSE)

# --- Main-figure Panel B: two panels (DNN vs Age Control marg, 4 estimators) ---
no_ctrl_main <- no_ctrl %>% filter(variant == "primary")  # base_full only
pB_left_main  <- make_auc_panel_simple(
  no_ctrl_main, "DNN (Baseline)",
  hue = OI_VERMILLION,
  show_y = TRUE, show_x = TRUE, annotate_ref = FALSE, compact = TRUE
)

# Build 4-method data frame for the right plot:
#   1. No sample split  (current run, nosplit_age, marg AUC)
#   2. Sample split     (current run, refit_age, A-side, marg AUC)
#   3. Cross-fit (K=2)  (current run, crossfit_age, marg AUC)
#   4. Cross-fit (K=3)  (k3 sweep, crossfit_k3_age, marg AUC)
# All four are AUC_marg ("controlled" / marginalized) on the balanced test set.
age_ctrl_compare <- bind_rows(
  no_split_df %>%
    select(coef, fold, auc, training) %>%
    mutate(estimator = "No sample split"),
  crossfit_df %>%
    filter(method == "refit_age", marg == "controlled") %>%
    select(coef, fold, auc, training) %>%
    mutate(estimator = "Sample split"),
  crossfit_df %>%
    filter(method == "crossfit_age", marg == "controlled") %>%
    select(coef, fold, auc, training) %>%
    mutate(estimator = "Cross-fit (K=2)"),
  k3_df %>%
    filter(method == "crossfit_k3_age", marg == "controlled") %>%
    select(coef, fold, auc, training) %>%
    mutate(estimator = "Cross-fit (K=3)")
) %>%
  mutate(estimator = factor(
    estimator,
    levels = c("No sample split", "Sample split", "Cross-fit (K=2)", "Cross-fit (K=3)")
  ))

estimator_hues <- c(
  "No sample split" = GREY_NOSPLIT,
  "Sample split"    = SKY_PALE,
  "Cross-fit (K=2)" = OI_SKY,
  "Cross-fit (K=3)" = OI_BLUE_DARK,
  "Joint model"     = OI_ORANGE
)

make_auc_panel_4 <- function(df, title, show_y = TRUE, show_x = TRUE,
                             annotate_ref = FALSE, compact = TRUE) {
  y_label <- if (show_y) {
    if (compact) "AUC (balanced test)" else "AUC on balanced test data"
  } else NULL
  p <- ggplot(df, aes(x = training, y = auc, fill = estimator)) +
    geom_hline(yintercept = ref_bal, linetype = "dashed",
               linewidth = 0.4, colour = GREY_REF) +
    geom_boxplot(aes(group = interaction(training, estimator)),
                 position = position_dodge(0.78),
                 outlier.shape = NA, width = 0.65, linewidth = 0.3,
                 alpha = 0.85, colour = GREY_REF) +
    geom_point(aes(group = interaction(training, estimator)),
               position = position_jitterdodge(
                 dodge.width = 0.78, jitter.width = 0.08, seed = 1),
               size = 0.55, shape = 21, colour = GREY_REF, stroke = 0.15) +
    scale_fill_manual(values = estimator_hues, name = NULL) +
    scale_y_continuous(breaks = seq(0.5, 0.8, 0.1)) +
    coord_cartesian(ylim = y_range) +
    labs(title = title, x = NULL, y = y_label) +
    shared_theme +
    theme(
      panel.grid.minor = element_blank(),
      legend.position  = c(0.02, 0.02),
      legend.justification = c(0, 0),
      legend.key.size  = unit(0.08, "in"),
      legend.text      = element_text(size = 6, family = "serif"),
      legend.margin    = margin(0, 2, 0, 2),
      legend.background = element_rect(fill = alpha("white", 0.7), colour = NA)
    )
  if (!show_y) {
    p <- p + theme(axis.text.y = element_blank(),
                   axis.ticks.y = element_blank())
  }
  if (!show_x) {
    p <- p + theme(axis.text.x = element_blank(),
                   axis.ticks.x = element_blank())
  } else if (compact) {
    p <- p + theme(axis.text.x = element_text(angle = 25, hjust = 1))
  }
  if (annotate_ref) {
    p <- p + annotate("text",
                      x = Inf, y = ref_bal,
                      hjust = 1.02, vjust = -0.4,
                      label = "Unconfounded target",
                      size = 2.2, family = "serif", colour = GREY_REF)
  }
  p
}

pB_right_main <- make_auc_panel_4(
  age_ctrl_compare, "DNN w. Age Control (marginalised)",
  show_y = FALSE, show_x = TRUE, annotate_ref = TRUE, compact = TRUE
)

# Other methods with age as Z, confounded training only, AUC marginalised over
# the training age distribution (joint_results.csv from train_joint.py,
# competitor_results.csv from competitors_from_checkpoints.py).
competitor_labels <- c(
  joint_age                = "Joint DNN",
  posthoc_web_age          = "DNN (Baseline) + Orth.",
  posthoc_web_features_age = "PHO + Refit [48]",
  leace_age                = "LEACE [49]",
  dfr_val_age              = "DFR [31]"
)
competitor_df <- bind_rows(
  read.csv(paste0(RUN_DIR, "joint_results.csv")),
  read.csv(paste0(RUN_DIR, "competitor_results.csv"))
) %>%
  filter(method %in% names(competitor_labels))
pB_comp_main <- make_auc_panel_simple(
  competitor_df %>%
    filter(coef == 2) %>%
    mutate(training = factor(competitor_labels[method], levels = competitor_labels),
           auc = auc_marg),
  "Others (Confounded Training)", hue = OI_ORANGE,
  show_y = FALSE, show_x = TRUE, compact = TRUE
)

# ---------------------------------------------------------------------------
# Assemble with grid viewports
# ---------------------------------------------------------------------------
# --- Main paper figure: Panel A | Panel B side-by-side (double-column) -----
# Outer layout: 1 row × 2 cols (A | B). Panel B holds pB_left_main (DNN (Baseline)),
# pB_right_main (DNN w. Age Control) and pB_comp_main (others, confounded training).
cairo_pdf(OUT_PATH_MAIN, width = 7.16, height = 2.0, pointsize = 8, fallback_resolution = 600)
grid.newpage()

lay <- grid.layout(nrow = 1, ncol = 2,
                   widths = unit(c(0.34, 0.66), "npc"))
pushViewport(viewport(layout = lay))

pushViewport(viewport(layout.pos.row = 1, layout.pos.col = 1))
print(panel_A, newpage = FALSE, vp = viewport(width = 1, height = 1))
popViewport()

pushViewport(viewport(layout.pos.row = 1, layout.pos.col = 2,
                      layout = grid.layout(nrow = 1, ncol = 3,
                                           widths = unit(c(0.22, 0.44, 0.34), "npc"))))
print(pB_left_main,  newpage = FALSE, vp = viewport(layout.pos.col = 1))
print(pB_right_main, newpage = FALSE, vp = viewport(layout.pos.col = 2))
print(pB_comp_main,  newpage = FALSE, vp = viewport(layout.pos.col = 3))
popViewport()
popViewport()
invisible(dev.off())
message("Saved: ", OUT_PATH_MAIN)

# --- Appendix figure: No Control (full vs half) | β_age recovery -----------
cairo_pdf(OUT_PATH_APPENDIX, width = 7.16, height = 1.9, pointsize = 8, fallback_resolution = 600)
grid.newpage()
pushViewport(viewport(layout = grid.layout(nrow = 1, ncol = 2,
                                           widths = unit(c(0.5, 0.5), "npc"))))
print(pS1_nc,  newpage = FALSE, vp = viewport(layout.pos.col = 1))
print(pS2_age, newpage = FALSE, vp = viewport(layout.pos.col = 2))
popViewport()
invisible(dev.off())
message("Saved: ", OUT_PATH_APPENDIX)

# --- Lambda averages summary (per method × coef, mean ± sd over folds) -----
# Printed to console so we can paste numbers into the appendix prose.
if (exists("cf_raw")) {
  lam_summary <- cf_raw %>%
    filter(!is.na(lam)) %>%
    mutate(training = if_else(coef == 0, "Balanced", "Confounded")) %>%
    group_by(method, training) %>%
    summarize(lam_mean = mean(lam), lam_sd = sd(lam), .groups = "drop") %>%
    arrange(method, training)
  message("--- Selected lambda averages (over 5 folds) ---")
  print(lam_summary, n = Inf)
}

# --- Appendix table: test AUC per method, Z = age, mean (sd) over 5 folds ---
table_labels <- c(
  base_full                = "DNN (Baseline)",
  nosplit_age              = "DNN with Controls, no sample split",
  refit_age                = "DNN with Controls, sample split",
  crossfit_age             = "DNN with Controls, cross-fit (K=2)",
  crossfit_k3_age          = "DNN with Controls, cross-fit (K=3)",
  joint_age                = "Joint DNN",
  posthoc_web_age          = "DNN (Baseline) + Orth.",
  posthoc_web_features_age = "PHO + Refit",
  leace_age                = "LEACE",
  dfr_val_age              = "DFR"
)
table_df <- bind_rows(
  read.csv(paste0(RUN_DIR, "raw_results.csv")) %>% filter(method == "base_full"),
  read.csv(paste0(RUN_DIR, "competitor_results.csv")) %>% filter(method == "nosplit_age"),
  cf_raw, k3_raw, competitor_df
) %>%
  filter(method %in% names(table_labels)) %>%
  select(method, coef, fold, auc, auc_marg)
fmt <- function(x) if (all(is.na(x))) "--" else sprintf("%.3f (%.3f)", mean(x), sd(x))
table_rows <- table_df %>%
  group_by(method) %>%
  summarize(
    bal_auc  = fmt(auc[coef == 0]),      bal_marg  = fmt(auc_marg[coef == 0]),
    conf_auc = fmt(auc[coef == 2]),      conf_marg = fmt(auc_marg[coef == 2]),
    .groups = "drop"
  ) %>%
  mutate(method = factor(method, levels = names(table_labels))) %>%
  arrange(method)
writeLines(c(
  "\\begin{tabular}{lcccc}",
  "\\toprule",
  " & \\multicolumn{2}{c}{Balanced training} & \\multicolumn{2}{c}{Confounded training} \\\\",
  "Method & AUC & AUC (marg.) & AUC & AUC (marg.) \\\\",
  "\\midrule",
  sprintf("%s & %s & %s & %s & %s \\\\", table_labels[as.character(table_rows$method)],
          table_rows$bal_auc, table_rows$bal_marg, table_rows$conf_auc, table_rows$conf_marg),
  "\\bottomrule",
  "\\end{tabular}"
), paste0(OUT_DIR, "Tab_UKBB_competitors.tex"))
message("Saved: ", OUT_DIR, "Tab_UKBB_competitors.tex")
