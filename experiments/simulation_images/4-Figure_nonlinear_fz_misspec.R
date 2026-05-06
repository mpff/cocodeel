pdf(NULL)  # suppress automatic Rplots.pdf when run non-interactively
library(readr)
library(dplyr)
library(ggplot2)
library(tikzDevice)
library(latex2exp)
library(patchwork)

# Misspecification companion to 4-Figure_nonlinear_fz.R: same layout
# (single method, fx top, fz bottom, MSPE | Bias² | Var), but reads
# from `nonlinear_fz_misspec.csv` where the model is fed the raw 1-d
# covariate Z (no spline basis). The MSPE values are roughly two
# orders of magnitude larger than the well-specified case, so the
# y-axis is widened to c(1e-4, 10).

df <- read_csv("results/simulation_images/nonlinear_fz_misspec.csv") %>%
  filter(model == "posthoc_xfit") %>%
  mutate(effect = factor(effect, levels = c("y", "fx", "fr", "fz")))

shared_theme <- theme_bw() +
  theme(
    legend.title.position = "left",
    legend.key.height = unit(0.06, 'in'),
    legend.key.width = unit(0.15, 'in'),
    legend.ticks.length = unit(c(-.05, 0), 'in'),
    legend.ticks = element_line(color = 'black'),
    legend.direction = "horizontal",
    legend.margin = margin(0, unit = "inch"),
    legend.background = element_rect(color = NA, fill = NA),
    axis.text.x = element_text(angle = 30, hjust = 1, vjust = 1.2, size = 6),
    axis.text.y = element_text(hjust = 1.25, size = 6),
    axis.title.x = element_text(margin = margin(t = 0)),
    text = element_text(size = 8, family = "serif"),
    plot.margin = margin(2, 2, 2, 2)
  )

# Single shared colour-scale object so patchwork can dedupe (and so we
# can reuse the well-specified figure's bz colourbar conventions).
BZ_COLOR_SCALE <- scale_color_viridis_c(
  begin = 0, end = 1, option = "viridis",
  limits = c(-0.025, 4.25), breaks = c(0, 1, 2, 3, 4),
  name = expression(beta[Z])
)

make_panel <- function(data, ylab, ylim = c(1e-4, 10)) {
  ggplot(data, aes(x = n * 0.5, y = value, group = bz)) +
    geom_line(aes(color = bz), alpha = 0.8, linewidth = 0.8) +
    geom_point(aes(color = bz), alpha = 0.8, size = 0.8) +
    scale_x_log10(
      name = TeX("$N_{train}$ ($\\log_{10}$ scale)"),
      breaks = 100 * 2^(1:7)
    ) +
    scale_y_log10(name = ylab) +
    BZ_COLOR_SCALE +
    coord_cartesian(
      xlim = c(175, 100 * 2^7.05),
      ylim = ylim
    ) +
    shared_theme
}

YLIM <- c(1e-4, 10)

# Top row — fx (direct image effect): MSPE, Bias², Var.
top_mspe <- make_panel(
  df %>% filter(effect == "fx", metric == "mspe"),
  ylab = TeX("$MSPE(\\hat{f}_X)$"), ylim = YLIM
)
top_bias <- make_panel(
  df %>% filter(effect == "fx", metric == "bias2"),
  ylab = TeX("$Bias^2(\\hat{f}_X)$"), ylim = YLIM
)
top_var <- make_panel(
  df %>% filter(effect == "fx", metric == "var"),
  ylab = TeX("$Var(\\hat{f}_X)$"), ylim = YLIM
)

# Bottom row — fz (nonlinear covariate effect): MSPE, Bias², Var.
bot_mspe <- make_panel(
  df %>% filter(effect == "fz", metric == "mspe"),
  ylab = TeX("$MSPE(\\hat{f}_Z)$"), ylim = YLIM
)
bot_bias <- make_panel(
  df %>% filter(effect == "fz", metric == "bias2"),
  ylab = TeX("$Bias^2(\\hat{f}_Z)$"), ylim = YLIM
)
bot_var <- make_panel(
  df %>% filter(effect == "fz", metric == "var"),
  ylab = TeX("$Var(\\hat{f}_Z)$"), ylim = YLIM
)

# Single legend INSIDE the top-left panel (MSPE(fx)) at bottom-left,
# matching the well-specified figure's convention. Suppress on others.
top_mspe <- top_mspe +
  theme(legend.position = c(0.32, 0.93),
        legend.direction = "horizontal",
        legend.key.width = unit(0.15, 'in'),
        legend.background = element_rect(color = NA, fill = NA))
top_bias <- top_bias + theme(legend.position = "none")
top_var  <- top_var  + theme(legend.position = "none")
bot_mspe <- bot_mspe + theme(legend.position = "none")
bot_bias <- bot_bias + theme(legend.position = "none")
bot_var  <- bot_var  + theme(legend.position = "none")

fig <- (top_mspe | top_bias | top_var) /
       (bot_mspe | bot_bias | bot_var)

ggsave("graphics/Fig_nonlinear_fz_misspec.pdf", fig,
       width = 6.0, height = 2.7, units = "in",
       dpi = 600, device = cairo_pdf)
