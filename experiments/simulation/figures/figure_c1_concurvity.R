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

# Grouped by how the method is fit: end to end with Z by SGD, by post-hoc
# orthogonalisation of a fit, or by the refit on held-out data.
# Citation numbers follow the preprint's bibliography.
model_groups <- list(
  c(nam = "DNN with Controls (SGD)",
    nam_mlp = "NAM (MLP fz, [8])",
    nam_mlp_conc = "+ concurvity reg. [17]"),
  c(ssn = "SSN [9]",
    posthoc_web_features = "PHO + Refit [48]"),
  c(refit = "DNN with Controls",
    refit_orth = "DNN with Controls\n+ Orthogonalisation"))
model_labels <- unlist(unname(model_groups))

df_bz <- df_raw %>%
  filter(model %in% names(model_labels)) %>%
  mutate(model = factor(model_labels[model], levels = model_labels),
         effect = factor(effect, levels = c('y', 'fx', 'fr', 'fz')))

# Okabe-Ito colour per fitted model: an orthogonalised method shares the colour of
# the fit it corrects (SSN corrects DNN with Controls (SGD)) and is drawn with
# diamonds, all others with circles; dashes mark the methods whose target is f_X^re
method_colors <- setNames(c(
  "#CC79A7", "#E69F00", "#D55E00",   # DNN with Controls (SGD), NAM, + concurvity reg.
  "#CC79A7", "#009E73",              # SSN, PHO + Refit
  "#0072B2", "#0072B2"               # DNN with Controls, + Orthogonalisation
), model_labels)
orthogonalised <- model_labels %in% c("SSN [9]", "PHO + Refit [48]", "DNN with Controls\n+ Orthogonalisation")
method_shapes <- setNames(ifelse(orthogonalised, 23, 21), model_labels)
targets_fr <- orthogonalised | model_labels == "+ concurvity reg. [17]"
method_lines  <- setNames(ifelse(targets_fr, "22", "solid"), model_labels)

# blank legend rows between the groups
spacers <- paste0("spacer", seq_len(length(model_groups) - 1))
legend_breaks <- head(unlist(Map(function(g, s) c(unname(g), s),
                                 model_groups, c(spacers, NA))), -1)
spacer_values <- function(v) setNames(rep(v, length(spacers)), spacers)
legend_colors <- c(method_colors, spacer_values("transparent"))
legend_shapes <- c(method_shapes, spacer_values(NA))
legend_lines  <- c(method_lines, spacer_values("blank"))
legend_labels <- setNames(ifelse(legend_breaks %in% spacers, " ", legend_breaks), legend_breaks)

# Shared theme — legend is collected by patchwork and placed at the bottom
# (see plot_layout(guides = "collect") below). In-panel placement was
# unreadable: 7 vertically-stacked entries overlapped data curves.
shared_theme <- theme_bw() +
  theme(
    legend.title = element_blank(),
    legend.key.height = unit(0.06, 'in'),
    legend.key.width = unit(0.3, 'in'),
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
make_plot <- function(data, ylab, strip_labels = TRUE, ylim = c(0.3, 2e-4)) {
  p <- ggplot(
    data,
    aes(x = n * 0.5, y = value, group = model)
  ) +
    geom_line(aes(color = model, linetype = model), alpha = 0.85, linewidth = 0.7) +
    geom_point(aes(color = model, fill = model, shape = model), alpha = 0.85, size = 1.1, stroke = 0.5) +
    scale_x_log10(
      name = TeX("$N_{train}$ ($\\log_{10}$ scale)"),
      breaks = 100 * 2^(1:9)
    ) +
    scale_y_log10(name = ylab, labels = function(x) formatC(x, format = "e", digits = 0)) +
    scale_color_manual(name = NULL, values = legend_colors, limits = legend_breaks,
                       breaks = legend_breaks, labels = legend_labels) +
    scale_fill_manual(name = NULL, values = legend_colors, limits = legend_breaks,
                      breaks = legend_breaks, labels = legend_labels) +
    scale_shape_manual(name = NULL, values = legend_shapes, limits = legend_breaks,
                       breaks = legend_breaks, labels = legend_labels) +
    scale_linetype_manual(name = NULL, values = legend_lines, limits = legend_breaks,
                          breaks = legend_breaks, labels = legend_labels) +
    guides(color = guide_legend(ncol = 1)) +
    coord_cartesian(
      xlim = c(175, 100 * 2^9.5),
      ylim = ylim
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
  theme(legend.position = "right", plot.margin = margin(0, 6, 0, 0))

ggsave("experiments/simulation/output/graphics/Fig2_Concurvity.pdf", b, width = 5.5, height = 1.45, units = "in", dpi=600, device = cairo_pdf)

# Appendix: the Bias^2 + Var split of both MSPE panels (rows), on one y-range
ylim_bv <- c(0.3, 3e-6)
bv <- list(
  make_plot(df_bz %>% filter(effect == "fx", metric == "bias2"), TeX("$Bias^2(\\hat{f}_X)$"), ylim = ylim_bv),
  make_plot(df_bz %>% filter(effect == "fr", metric == "bias2"), TeX("$Bias^2(\\hat{f}^{re}_X)$"), ylim = ylim_bv),
  make_plot(df_bz %>% filter(effect == "fx", metric == "var"), TeX("$Var(\\hat{f}_X)$"), ylim = ylim_bv),
  make_plot(df_bz %>% filter(effect == "fr", metric == "var"), TeX("$Var(\\hat{f}^{re}_X)$"), ylim = ylim_bv))
b_bv <- wrap_plots(bv, ncol = 2) +
  plot_layout(guides = "collect") &
  theme(legend.position = "right", plot.margin = margin(2, 6, 2, 0))

ggsave("experiments/simulation/output/graphics/Fig_concurvity_bias_var.pdf", b_bv, width = 5.5, height = 2.9, units = "in", dpi=600, device = cairo_pdf)
