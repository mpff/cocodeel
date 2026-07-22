# Temporary side figure: Fig 2 with the source-protocol CF-Net spliced in
# (side_aggregate_cfnet.py output). Delete together with the side scripts.
pdf(NULL)  # suppress automatic Rplots.pdf when run non-interactively
library(readr)
library(dplyr)
library(ggplot2)
library(tikzDevice)
library(latex2exp)
library(png)
library(grid)
library(patchwork)

effect_names <- c(
  'y' = 'y',
  'fx' = 'Image Effect',
  'fr' = 'Residual Image Effect',
  'fz' = 'Covariate Effect'
)

df_raw <- read_csv("experiments/simulation/output/side_concurvity_fixed_cfnet.csv")

# Of the three Siems penalty strengths, only the best is plotted — all
# would drown the panel. Best = lowest mean log10 MSPE(f_X) across n, the
# metric of the left panel. CF-Net is pinned to its as-published strength:
# its three strengths are indistinguishable on that score (spread < 0.01),
# so a data-driven pick would be a coin flip.
pick_best <- function(df, prefix) {
  df %>%
    filter(startsWith(model, prefix), effect == "fx", metric == "mspe") %>%
    group_by(model) %>%
    summarise(score = mean(log10(value)), .groups = "drop") %>%
    slice_min(score, n = 1, with_ties = FALSE) %>%
    pull(model)
}
best_siems <- pick_best(df_raw, "nam_mlp_conc_")
best_cfnet <- "cfnet_1"
siems_lam <- sub("nam_mlp_conc_", "", best_siems)
cfnet_lam <- sub("cfnet_", "", best_cfnet)

model_levels <- c("nam", "nam_mlp", best_siems, "ssn", "posthoc_web",
                  best_cfnet, "refit", "refit_orth")
model_labels <- c(
  "NAM (lin. fz) [7]",
  "NAM [7]",
  sprintf("NAM + Concurvity Reg. (%s) [20]", siems_lam),
  "SSN [8]",
  "DNN (Baseline) + Orth. [17]",
  sprintf("CF-Net (%s)", cfnet_lam),
  "DNN with Controls",
  "DNN with Controls\n+ Orthogonalisation")

df_bz <- df_raw %>%
  mutate(model = factor(model, levels = model_levels, labels = model_labels)) %>%
  filter(!is.na(model)) %>%
  mutate(effect = factor(effect, levels=c('y', 'fx', 'fr', 'fz')))

# Manual color scale.
#
# NAM family → viridis cool ramp (purple → blue → teal); position encodes
# regularisation strength. SSN → plasma magenta — a different post-hoc
# approach, off the viridis path.
#
# DNN w. Controls and DNN w. Controls + Orth. mirror Fig 1bz's two
# palettes (viridis / magma) at the position corresponding to the bz
# value used in this simulation block (bz = 1), under the same scale
# limits as Fig 1's scale_color_viridis_c (c(-0.025, 4.25)). Computing
# the colours programmatically (rather than hardcoding hex) keeps the
# two figures visually linked: the same bz value reads as the same
# colour in both.
FIG1_BZ_LIMITS <- c(-0.025, 4.25)
FIG1_BZ_ANCHOR <- 2.5  # bz value used to pick the highlight colour;
                       # 2.5 lands in the warm-mid range of viridis/magma,
                       # giving teal-green + coral-pink (both readable and
                       # CB-distinct from the cool NAM ramp).
fig1_bz_pos <- (FIG1_BZ_ANCHOR - FIG1_BZ_LIMITS[1]) /
               (FIG1_BZ_LIMITS[2] - FIG1_BZ_LIMITS[1])

bz_anchor_color <- function(option) {
  scales::viridis_pal(begin = fig1_bz_pos, end = fig1_bz_pos,
                      option = option)(1)
}

method_colors <- setNames(c(
  "#440154",                    # NAM — viridis(0.00), deep purple
  "#3B528B",                    # NAM-MLP — viridis(0.25), blue
  "#26828E",                    # NAM-MLP + Reg. — viridis(0.50), teal
  "#9C179E",                    # SSN — plasma(0.40), magenta
  "#7F7F7F",                    # Weber post-hoc — neutral grey
  "#ED7953",                    # CF-Net — plasma(0.70), coral
  bz_anchor_color("viridis"),   # DNN with Controls
  bz_anchor_color("magma")      # DNN with Controls + Orth.
), model_labels)



# Shared theme — legend is collected by patchwork and placed at the bottom
# (see plot_layout(guides = "collect") below). In-panel placement was
# unreadable: 7 vertically-stacked entries overlapped data curves.
shared_theme <- theme_bw() +
  theme(
    legend.title = element_blank(),
    legend.key.height = unit(0.07, 'in'),
    legend.key.width = unit(0.14, 'in'),
    legend.background = element_rect(color = NA, fill = NA),
    legend.text = element_text(size = 6, family = "serif",
                               margin = margin(l = 1, r = 6)),
    legend.margin = margin(0, 0, 0, 0),
    legend.box.margin = margin(-4, 0, 0, 0),
    legend.box.spacing = unit(0, "pt"),
    legend.spacing.x = unit(0, "pt"),
    legend.spacing.y = unit(0, "pt"),
    axis.text.x = element_text(angle = 30, hjust = 1, vjust = 1.2, size=6),
    axis.text.y = element_text(hjust = 1.25, size=6),
    axis.title.x = element_text(margin = margin(t = 0)),
    text = element_text(size = 8, family = "serif"),
    plot.margin = margin(2, 2, 2, 2)
  )

# Reusable plotting function
make_plot <- function(data, ylab, strip_labels = TRUE) {
  p <- ggplot(
    data,
    aes(x = n * 0.5, y = value, group = model)
  ) +
    geom_line(aes(color = model), alpha = 0.8, linewidth = 0.8) +
    geom_point(aes(color = model), alpha = 0.8, size = 0.8) +
    scale_x_log10(
      name = TeX("$N_{train}$ ($\\log_{10}$ scale)"),
      breaks = 100 * 2^(1:9)
    ) +
    scale_y_log10(name = ylab) +
    scale_color_manual(name = NULL, values = method_colors,
                       breaks = names(method_colors)) +
    guides(color = guide_legend(ncol = 1)) +
    coord_cartesian(
      xlim = c(175, 100 * 2^9.5),
      ylim = c(0.5, 1e-4)
    ) +
    shared_theme

  if (strip_labels) {
    p <- p + theme(strip.text.y = element_blank())
  }

  p
}

# Build plots
METHODS <- model_labels

b1 <- make_plot(
  df_bz %>% filter(effect == "fx", metric == "mspe", model %in% METHODS),
  ylab = TeX("$MSPE(\\hat{f}_X)$"),
  strip_labels = FALSE
)

b2 <- make_plot(
  df_bz %>% filter(effect == "fr", metric == "mspe", model %in% METHODS),
  ylab = TeX("$MSPE(\\hat{f}^{re}_X)$"),
  strip_labels = FALSE
)

# Collect the (identical) legends from both panels and place at right
# (single vertical column). Figure is double-column width.
b <- (b1 + b2) +
  plot_layout(guides = "collect") &
  theme(legend.position = "right")

ggsave("experiments/simulation/output/graphics/side_Fig2_Concurvity_fixed_cfnet.pdf", b, width = 5.0, height = 1.5, units = "in", dpi=600, device = cairo_pdf)

