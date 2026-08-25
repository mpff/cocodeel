# Study D: estimation under a noisily measured control variable.
# Layout mirrors figure_a1 (Fig 1b): x = N_train (log10), colour gradient over
# the corruption level a of Z_tilde = (1-a) Z + a eps, facet rows = method;
# left block targets f_X (refit, base), right block f_X^re (refit_orth, Weber).
# One PDF per metric: MSPE, Bias^2, Var.
library(readr)
library(dplyr)
library(ggplot2)
library(latex2exp)
library(patchwork)

effect_names <- c(
  'y' = 'y',
  'fx' = 'Image Effect',
  'fr' = 'Residual Image Effect',
  'fz' = 'Covariate Effect'
)

df_a <- read_csv("experiments/simulation/output/increasing_a.csv") %>%
  mutate(model = factor(
    model,
    levels = c("refit", "refit_orth", "base", "posthoc_web"),
    labels = c(
      "DNN with Controls",
      "DNN with Controls\n+ Orthogonalisation",
      "DNN (Baseline)",
      "[17] DNN (Baseline)\n+ Orthogonalisation")
  )) %>%
  filter(!is.na(model)) %>%
  mutate(effect = factor(effect, levels = c('y', 'fx', 'fr', 'fz')))


### PLOT ###

# Shared theme
shared_theme <- theme_bw() +
  theme(
    legend.title.position = "left",
    legend.key.height = unit(0.06, 'in'),
    legend.key.width = unit(0.15, 'in'),
    legend.ticks.length = unit(c(-.05, 0), 'in'),
    legend.ticks = element_line(color = 'black'),
    legend.position = c(0.5, 0.9),
    legend.direction = "horizontal",
    legend.margin = margin(0, unit = "inch"),
    legend.background = element_rect(color = NA, fill = NA),
    axis.text.x = element_text(angle = 30, hjust = 1, vjust = 1.2, size = 6),
    axis.text.y = element_text(hjust = 1.25, size = 6),
    text = element_text(size = 8, family = "serif"),
    plot.margin = margin(2, 2, 2, 2)
  )

# Reusable plotting function
make_plot <- function(data, ylab, color_option, show_legend = TRUE,
                      ylim = c(.4 * 1e-4, 1.25), xlim = c(175, 100 * 2^7.05),
                      legend.pos = c(0.5, 0.9)) {
  ggplot(data, aes(x = n * 0.5, y = value, group = a)) +
    geom_line(aes(color = a), alpha = 0.8, linewidth = 0.8) +
    geom_point(aes(color = a), alpha = 0.8, size = 0.8) +
    scale_x_log10(
      name = TeX("$N_{train}$ ($\\log_{10}$ scale)"),
      breaks = 100 * 2^(1:7)
    ) +
    scale_y_log10(name = ylab) +
    scale_color_viridis_c(
      begin = 0, end = 1, option = color_option,
      limits = c(-0.006, 1.006),
      breaks = c(0, 0.25, 0.5, 0.75, 1),
      name = TeX("$a$")
    ) +
    coord_cartesian(xlim = xlim, ylim = ylim) +
    facet_grid(model ~ .) +
    shared_theme +
    theme(legend.position = if (show_legend) legend.pos else "none")
}

plot_params <- list(
  list(metric = "mspe",  fx_lab = "$MSPE(\\hat{f}_X)$",   fr_lab = "$MSPE(\\hat{f}^{re}_X)$",   out = "FigD1_a_mspe.pdf",  ylim = c(.4 * 1e-3, 1.25)),
  list(metric = "bias2", fx_lab = "$Bias^2(\\hat{f}_X)$", fr_lab = "$Bias^2(\\hat{f}^{re}_X)$", out = "FigD1_a_bias2.pdf", ylim = c(.4 * 1e-4, 1.25)),
  list(metric = "var",   fx_lab = "$Var(\\hat{f}_X)$",    fr_lab = "$Var(\\hat{f}^{re}_X)$",    out = "FigD1_a_var.pdf",   ylim = c(.4 * 1e-4, 1.25))
)

for (m in plot_params) {
  p_fx <- make_plot(
    df_a %>% filter(effect == "fx", metric == m$metric,
                    model %in% c("DNN (Baseline)", "DNN with Controls")),
    ylab = TeX(paste(m$fx_lab, " ($\\log_{10}$ scale)")),
    color_option = "viridis", ylim = m$ylim
  )
  p_fr <- make_plot(
    df_a %>% filter(effect == "fr", metric == m$metric,
                    model %in% c("DNN with Controls\n+ Orthogonalisation",
                                 "[17] DNN (Baseline)\n+ Orthogonalisation")),
    ylab = TeX(paste(m$fr_lab, " ($\\log_{10}$ scale)")),
    color_option = "inferno", ylim = m$ylim
  )
  ggsave(file.path("experiments/simulation/output/graphics", m$out), p_fx + p_fr,
         width = 4.16, height = 2.4, units = "in", dpi = 600, device = cairo_pdf)
  cat("saved", m$out, "\n")
}
