# UKBB study arms — does the control still work when the additive assumptions are broken?
#
# Arm A: sex drawn independently of age, no interaction (the Section 6 study).
# Arm B: sex correlated with age.
# Arm C: sex correlated with age, plus a signal-confounder interaction.
#
# Run from project root:
#   Rscript experiments/ukbb/figures/UKBB_arms_fig.R

library(ggplot2)
library(dplyr)

RUNS <- "experiments/ukbb/runs/"
OUT_DIR <- paste0(RUNS, "graphics_arms/")
if (!dir.exists(OUT_DIR)) dir.create(OUT_DIR, recursive = TRUE)

# true age coefficient in raw logit units; unchanged across arms because the
# interaction is orthogonal to the additive span (binary sex, symmetric age)
TRUE_B_AGE <- -0.298

ARMS <- c(
  "final_v2"           = "rho == 0 * ',' ~ beta[int] == 0",
  "nonadd_int0_rho0.4" = "rho == 0.2 * ',' ~ beta[int] == 0",
  "nonadd_int2_rho0.4" = "rho == 0.2 * ',' ~ beta[int] == 2"
)

# Okabe-Ito, matching UKBB_application_fig.R so colours mean the same thing across figures
METHODS <- c(
  "base_full"        = "No control",
  "refit_age"        = "Age (split)",
  "crossfit_age"     = "Age (cross-fit)",
  "crossfit_age_sex" = "Age + sex (cross-fit)"
)
PALETTE <- c(
  "No control"            = "#D55E00",
  "Age (split)"           = "#009E73",
  "Age (cross-fit)"       = "#56B4E9",
  "Age + sex (cross-fit)" = "#CC79A7"
)

# ── load ─────────────────────────────────────────────────────────────────────
# refit_* live in raw_results, crossfit_* in crossfit_results; reading each from
# one file only avoids double-counting the single-split rows, which appear in both
read_arm <- function(run) {
  raw <- read.csv(paste0(RUNS, run, "_refit/raw_results.csv"))
  cf <- read.csv(paste0(RUNS, run, "_refit/crossfit_results.csv"))
  bind_rows(
    raw %>% filter(method %in% c("base_full", "refit_age")),
    cf %>% filter(method %in% c("crossfit_age", "crossfit_age_sex"))
  ) %>% mutate(arm = ARMS[[run]])
}

present <- names(ARMS)[dir.exists(paste0(RUNS, names(ARMS), "_refit"))]
df <- bind_rows(lapply(present, read_arm)) %>%
  # the uncontrolled model has no marginalised prediction
  mutate(
    metric = ifelse(method == "base_full", auc, auc_marg),
    method = factor(METHODS[method], levels = METHODS),
    arm = factor(arm, levels = unname(ARMS[present])),
    condition = factor(ifelse(coef == 0, "beta[age] == 0", "beta[age] == -2"),
                       levels = c("beta[age] == 0", "beta[age] == -2"))
  )

summarise_by <- function(d, col) {
  d %>%
    group_by(arm, method, condition) %>%
    summarise(mu = mean(.data[[col]], na.rm = TRUE),
              se = sd(.data[[col]], na.rm = TRUE) / sqrt(sum(!is.na(.data[[col]]))),
              n = sum(!is.na(.data[[col]])), .groups = "drop")
}

shared_theme <- theme_bw(base_size = 9) +
  theme(
    text = element_text(family = "serif"),
    panel.grid.minor = element_blank(),
    strip.background = element_blank(),
    legend.position = "top",
    legend.title = element_blank(),
    legend.margin = margin(b = -4)
  )

# ── panel 1: AUC, slope shows the confounding penalty ────────────────────────
auc <- summarise_by(df, "metric")
p_auc <- ggplot(auc, aes(condition, mu, colour = method, group = method)) +
  geom_line(linewidth = 0.4) +
  geom_errorbar(aes(ymin = mu - se, ymax = mu + se), width = 0.12, linewidth = 0.4) +
  geom_point(size = 1.4) +
  facet_wrap(~arm, nrow = 1, labeller = label_parsed) +
  scale_colour_manual(values = PALETTE) +
  scale_x_discrete(labels = scales::parse_format()) +
  labs(x = NULL, y = expression(AUC)) +
  shared_theme
ggsave(paste0(OUT_DIR, "Fig_UKBB_arms_auc.pdf"), p_auc,
       width = 5.5, height = 2.3, device = cairo_pdf)

# ── panel 2: recovery of the age coefficient under confounding ───────────────
bage <- df %>%
  filter(coef == 2, method != "No control") %>%
  summarise_by("b_age")
p_bage <- ggplot(bage, aes(method, mu, colour = method)) +
  geom_hline(yintercept = TRUE_B_AGE, linetype = "dashed", colour = "grey30", linewidth = 0.3) +
  geom_errorbar(aes(ymin = mu - se, ymax = mu + se), width = 0.15, linewidth = 0.4) +
  geom_point(size = 1.4) +
  facet_wrap(~arm, nrow = 1, labeller = label_parsed) +
  scale_colour_manual(values = PALETTE, guide = "none") +
  labs(x = NULL, y = expression(hat(beta)[age])) +
  shared_theme +
  theme(axis.text.x = element_text(angle = 30, hjust = 1, size = 6))
ggsave(paste0(OUT_DIR, "Fig_UKBB_arms_bage.pdf"), p_bage,
       width = 5.5, height = 2.5, device = cairo_pdf)

cat("arms plotted:", paste(present, collapse = ", "), "\n")
cat("wrote", OUT_DIR, "\n")
print(auc %>% arrange(arm, method, condition), n = 40)
