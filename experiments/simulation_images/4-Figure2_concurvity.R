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

df_bz <- read_csv("results/simulation_images/concurvity.csv") %>%
  mutate(model = factor(
    model,
    levels = c("covar", "covar_conc_0.1", "covar_conc_1", "covar_conc_10",
               "ssn", "posthoc_xfit", "posthoc_orth_xfit"),
    labels = c(
      "NAM [7]",
      "NAM + Reg. (0.1) [20]",
      "NAM + Reg. (1) [20]",
      "NAM + Reg. (10) [20]",
      "SSN [8]",
      "DNN w. Controls",
      "DNN w. Controls + Orth.")
  )) %>%
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

method_colors <- c(
  "NAM [7]"                 = "#440154",  # viridis(0.00) — deep purple
  "NAM + Reg. (0.1) [20]"   = "#3B528B",  # viridis(0.25) — blue
  "NAM + Reg. (1) [20]"     = "#287C8E",  # viridis(0.40) — teal-blue
  "NAM + Reg. (10) [20]"    = "#26828E",  # viridis(0.50) — teal
  "SSN [8]"                 = "#9C179E",  # plasma(0.40)  — magenta
  "DNN w. Controls"         = bz_anchor_color("viridis"),
  "DNN w. Controls + Orth." = bz_anchor_color("magma")
)



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
      breaks = 100 * 2^(0:9)
    ) +
    scale_y_log10(name = ylab) +
    scale_color_manual(name = NULL, values = method_colors,
                       breaks = names(method_colors)) +
    guides(color = guide_legend(ncol = 1)) +
    coord_cartesian(
      xlim = c(100, 100 * 2^9.5),
      ylim = c(0.5, 1e-4)
    ) +
    shared_theme

  if (strip_labels) {
    p <- p + theme(strip.text.y = element_blank())
  }

  p
}

# Build plots
METHODS <- c("NAM [7]", "NAM + Reg. (0.1) [20]", "NAM + Reg. (1) [20]",
             "NAM + Reg. (10) [20]", "SSN [8]", "DNN w. Controls",
             "DNN w. Controls + Orth.")

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

ggsave("graphics/Fig2_Concurvity.pdf", b, width = 5.0, height = 1.5, units = "in", dpi=600, device = cairo_pdf)

