# UKBB application figure (TEMP) — same layout as the paper version, but data
# from the NEW no-replace K=2 run (2026-05-03_16-43-32_n5k_noreplace_k2).
#
# Differences from the production script (UKBB_application_fig.R):
#   1. Panel B reads from one CSV (k2_crossfit_results.csv); no OLD_DIR
#      no-split fallback needed (no-samp now lives in the new CSV).
#   2. Right panel shows three estimators (No samp / Sample split /
#      Cross-fit K=2). No K=3 in this run.
#   3. Panel A densities reuse the OLD trainset_folds.csv — the resampling
#      DGP target distribution is the same; only with/without replacement
#      differs at the matching step, which doesn't change the marginal age
#      density of the resampled set in any visible way.
#
# Output: experiments/ukbb/runs/2026-05-03_16-43-32_n5k_noreplace_k2/graphics/
#         Fig_UKBB_application_main_noreplace.pdf
#
# Run from project root:
#   Rscript experiments/ukbb/figures/UKBB_application_fig_TEMP_noreplace.R

library(ggplot2); library(dplyr); library(tidyr); library(grid)

NEW_RUN  <- "experiments/ukbb/runs/2026-05-03_16-43-32_n5k_noreplace_k2/"
OLD_DATA <- "experiments/ukbb/runs/2026-04-26_13-16-37_final_v2/rexports/"

OUT_DIR  <- paste0(NEW_RUN, "graphics/")
if (!dir.exists(OUT_DIR)) dir.create(OUT_DIR, recursive = TRUE)
OUT_PATH <- paste0(OUT_DIR, "Fig_UKBB_application_main_noreplace.pdf")

# -- Okabe-Ito --
OI_BLUE       <- "#0072B2"
OI_ORANGE     <- "#E69F00"
OI_VERMILLION <- "#D55E00"
OI_SKY        <- "#56B4E9"
SKY_PALE      <- "#A8DBF0"
GREY_REF      <- "grey30"
GREY_NOSPLIT  <- "#BFBFBF"
DIST_HIGH <- OI_ORANGE
DIST_CTRL <- OI_BLUE

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

# ===========================================================================
# Panel A: densities (reuse OLD data; DGP shape governs this panel)
# ===========================================================================
actual <- read.csv(paste0(OLD_DATA, "testset.csv"))
folds  <- read.csv(paste0(OLD_DATA, "trainset_folds.csv"))
xlim_age <- range(c(actual$age, folds$age))

make_density <- function(df, title) {
  ggplot(df, aes(x = age, fill = factor(y))) +
    geom_density(alpha = 0.55, colour = NA, adjust = 1.2) +
    scale_fill_manual(values = c("0" = DIST_CTRL, "1" = DIST_HIGH),
                      guide = "none") +
    coord_cartesian(xlim = xlim_age, expand = FALSE) +
    labs(title = title, x = "Age (years)", y = NULL) +
    shared_theme +
    theme(
      axis.text.y        = element_blank(),
      axis.ticks.y       = element_blank(),
      panel.grid.minor   = element_blank(),
      panel.grid.major.y = element_blank(),
      plot.margin        = margin(1, 3, 1, 3),
      axis.title.x       = element_text(margin = margin(t = 1))
    )
}

g_orig <- ggplotGrob(make_density(actual,                       "Original"))
g_bal  <- ggplotGrob(make_density(folds %>% filter(coef == 0),  "Balanced"))
g_conf <- ggplotGrob(make_density(folds %>% filter(coef == 2),  "Confounded"))

panel_A <- ggplot() +
  coord_cartesian(xlim = c(0, 1), ylim = c(0, 1), expand = FALSE, clip = "off") +
  annotation_custom(g_orig, xmin = 0.04, xmax = 0.46, ymin = 0.55, ymax = 1.00) +
  annotation_custom(g_bal,  xmin = 0.04, xmax = 0.46, ymin = 0.00, ymax = 0.45) +
  annotation_custom(g_conf, xmin = 0.54, xmax = 0.96, ymin = 0.00, ymax = 0.45) +
  annotate("curve",
           x = 0.20, xend = 0.20, y = 0.55, yend = 0.46,
           curvature = 0.001,
           arrow = arrow(length = unit(0.05, "in"), type = "closed"),
           colour = GREY_REF, linewidth = 0.35) +
  annotate("text", x = 0.13, y = 0.50,
           label = 'beta["age"] == 0', parse = TRUE,
           size = 2.2, family = "serif", colour = GREY_REF) +
  annotate("curve",
           x = 0.43, xend = 0.55, y = 0.55, yend = 0.46,
           curvature = 0.35,
           arrow = arrow(length = unit(0.05, "in"), type = "closed"),
           colour = GREY_REF, linewidth = 0.35) +
  annotate("text", x = 0.62, y = 0.51,
           label = 'beta["age"] == -2.0', parse = TRUE,
           size = 2.2, family = "serif", colour = GREY_REF) +
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

# ===========================================================================
# Panel B: AUC from the new no-replace K=2 CSV
# ===========================================================================
res <- read.csv(paste0(NEW_RUN, "k2_crossfit_results.csv")) %>%
  mutate(training = factor(
    if_else(coef == 0, "Balanced training", "Confounded training"),
    levels = c("Balanced training", "Confounded training")
  ))

# DNN baseline AUC at coef=0 sets the unconfounded reference line.
ref_bal <- res %>%
  filter(method == "dnn", coef == 0) %>%
  pull(auc) %>% median()

y_range <- c(0.48, 0.82)

# -- Centre: No Control DNN (regular AUC; auc_marg is NaN for dnn) --
no_ctrl <- res %>% filter(method == "dnn") %>%
  select(coef, fold, auc, training)

pB_left <- ggplot(no_ctrl, aes(x = training, y = auc)) +
  geom_hline(yintercept = ref_bal, linetype = "dashed",
             linewidth = 0.4, colour = GREY_REF) +
  geom_boxplot(aes(group = training),
               fill = OI_VERMILLION, alpha = 0.85,
               width = 0.45, linewidth = 0.35, outlier.shape = NA,
               colour = GREY_REF) +
  geom_point(aes(group = training),
             position = position_jitter(width = 0.08, seed = 1),
             size = 0.8, shape = 21, fill = OI_VERMILLION, colour = GREY_REF,
             alpha = 0.85, stroke = 0.2) +
  scale_y_continuous(breaks = seq(0.5, 0.8, 0.1)) +
  coord_cartesian(ylim = y_range) +
  labs(title = "No Control (DNN)", x = NULL, y = "AUC (balanced test)") +
  shared_theme +
  theme(panel.grid.minor = element_blank(),
        axis.text.x = element_text(angle = 25, hjust = 1))

# -- Right: Age Control marg, 3 estimators --
age_ctrl <- bind_rows(
  res %>% filter(method == "posthoc_nosamp_age") %>%
    select(coef, fold, auc = auc_marg, training) %>%
    mutate(estimator = "No sample split"),
  res %>% filter(method == "posthoc_split_age") %>%
    select(coef, fold, auc = auc_marg, training) %>%
    mutate(estimator = "Sample split"),
  res %>% filter(method == "crossfit_k2_age") %>%
    select(coef, fold, auc = auc_marg, training) %>%
    mutate(estimator = "Cross-fit (K=2)")
) %>%
  mutate(estimator = factor(
    estimator,
    levels = c("No sample split", "Sample split", "Cross-fit (K=2)")
  ))

estimator_hues <- c(
  "No sample split" = GREY_NOSPLIT,
  "Sample split"    = SKY_PALE,
  "Cross-fit (K=2)" = OI_SKY
)

pB_right <- ggplot(age_ctrl, aes(x = training, y = auc, fill = estimator)) +
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
  labs(title = "Age Control (marg.)", x = NULL, y = NULL) +
  shared_theme +
  theme(
    panel.grid.minor  = element_blank(),
    legend.position   = c(0.02, 0.02),
    legend.justification = c(0, 0),
    legend.key.size   = unit(0.08, "in"),
    legend.text       = element_text(size = 6, family = "serif"),
    legend.margin     = margin(0, 2, 0, 2),
    legend.background = element_rect(fill = alpha("white", 0.7), colour = NA),
    axis.text.y       = element_blank(),
    axis.ticks.y      = element_blank(),
    axis.text.x       = element_text(angle = 25, hjust = 1)
  ) +
  annotate("text",
           x = Inf, y = ref_bal,
           hjust = 1.02, vjust = -0.4,
           label = "Unconfounded target",
           size = 2.2, family = "serif", colour = GREY_REF)

# ===========================================================================
# Assemble: A | (B_left | B_right)  side-by-side, IEEE double-column
# ===========================================================================
cairo_pdf(OUT_PATH, width = 7.16, height = 2.0, pointsize = 8,
          fallback_resolution = 600)
grid.newpage()
lay <- grid.layout(nrow = 1, ncol = 2, widths = unit(c(0.40, 0.60), "npc"))
pushViewport(viewport(layout = lay))

pushViewport(viewport(layout.pos.row = 1, layout.pos.col = 1))
print(panel_A, newpage = FALSE, vp = viewport(width = 1, height = 1))
popViewport()

pushViewport(viewport(layout.pos.row = 1, layout.pos.col = 2,
                      layout = grid.layout(nrow = 1, ncol = 2,
                                           widths = unit(c(1/3, 2/3), "npc"))))
print(pB_left,  newpage = FALSE, vp = viewport(layout.pos.col = 1))
print(pB_right, newpage = FALSE, vp = viewport(layout.pos.col = 2))
popViewport()
popViewport()
invisible(dev.off())
message("Saved: ", OUT_PATH)
