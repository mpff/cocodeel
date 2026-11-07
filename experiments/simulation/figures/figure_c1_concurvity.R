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

df_raw <- read_csv("experiments/simulation/output/concurvity.csv")

# Grouped by how the method is fit (Appendix, "Fitting routines"): jointly
# with Z, with a penalty, by post-hoc correction, or by the refit on held-out
# data. Citation numbers follow the preprint's bibliography.
model_groups <- list(
  c(nam = "Additive Model (SGD, like [8])"),
  c(nam_mlp_conc = "Additive Model + concurvity reg. [22]",
    cfnet = "CF-Net [41]"),
  c(ssn = "SSN [9]",
    posthoc_web_features = "PHO + Refit [19]"),
  c(refit = "DNN with Controls",
    refit_orth = "DNN with Controls\n+ Orthogonalisation"))
model_labels <- unlist(unname(model_groups))

df_bz <- df_raw %>%
  filter(model %in% names(model_labels)) %>%
  mutate(model = factor(model_labels[model], levels = model_labels),
         effect = factor(effect, levels = c('y', 'fx', 'fr', 'fz')))

# hue = fitting routine, lightness = member within routine
method_colors <- setNames(c(
  "#3F007D",              # joint fit with Z: purple
  "#7F3B08", "#E08214",   # penalty: oranges
  "#AE017E", "#F768A1",   # post-hoc correction: pinks
  "#00441B", "#41AB5D"    # refit on held-out data: greens
), model_labels)

# blank legend rows between the groups
spacers <- paste0("spacer", seq_len(length(model_groups) - 1))
legend_breaks <- head(unlist(Map(function(g, s) c(unname(g), s),
                                 model_groups, c(spacers, NA))), -1)
legend_colors <- c(method_colors, setNames(rep("transparent", length(spacers)), spacers))
legend_labels <- setNames(ifelse(legend_breaks %in% spacers, " ", legend_breaks), legend_breaks)

# Shared theme — legend is collected by patchwork and placed at the bottom
# (see plot_layout(guides = "collect") below). In-panel placement was
# unreadable: 7 vertically-stacked entries overlapped data curves.
shared_theme <- theme_bw() +
  theme(
    legend.title = element_blank(),
    legend.key.height = unit(0.06, 'in'),
    legend.key.width = unit(0.14, 'in'),
    legend.background = element_rect(color = NA, fill = NA),
    legend.text = element_text(size = 6, family = "serif",
                               margin = margin(l = 1, r = 6)),
    legend.margin = margin(0, 0, 0, 0),
    legend.box.margin = margin(0, 0, 0, 2),
    legend.box.spacing = unit(0, "pt"),
    legend.spacing.x = unit(0, "pt"),
    legend.spacing.y = unit(0, "pt"),
    axis.text.x = element_text(angle = 30, hjust = 1, vjust = 1.2, size=6),
    axis.text.y = element_text(hjust = 1.25, size=6),
    axis.title.x = element_text(margin = margin(t = 0)),
    text = element_text(size = 8, family = "serif"),
    plot.margin = margin(0, 0, 0, 0)
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
    scale_y_log10(name = ylab, labels = function(x) formatC(x, format = "e", digits = 0)) +
    scale_color_manual(name = NULL, values = legend_colors, limits = legend_breaks,
                       breaks = legend_breaks, labels = legend_labels) +
    guides(color = guide_legend(ncol = 1)) +
    coord_cartesian(
      xlim = c(175, 100 * 2^9.5),
      ylim = c(0.3, 2e-4)
    ) +
    shared_theme

  if (strip_labels) {
    p <- p + theme(strip.text.y = element_blank())
  }

  p
}

# Build plots
b1 <- make_plot(
  df_bz %>% filter(effect == "fx", metric == "mspe"),
  ylab = TeX("$MSPE(\\hat{f}_X)$"),
  strip_labels = FALSE
)

b2 <- make_plot(
  df_bz %>% filter(effect == "fr", metric == "mspe"),
  ylab = TeX("$MSPE(\\hat{f}^{re}_X)$"),
  strip_labels = FALSE
)

# Collect the (identical) legends from both panels and place at right
# (single vertical column). Figure is double-column width.
b <- (b1 + b2) +
  plot_layout(guides = "collect") &
  theme(legend.position = "right", plot.margin = margin(0, 0, 0, 0))

ggsave("experiments/simulation/output/graphics/Fig2_Concurvity.pdf", b, width = 5.5, height = 1.45, units = "in", dpi=600, device = cairo_pdf)

