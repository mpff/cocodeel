pdf(NULL)  # suppress automatic Rplots.pdf when run non-interactively
library(readr)
library(dplyr)
library(ggplot2)
library(tikzDevice)
library(latex2exp)
library(patchwork)

# Mirrors the bias-bz figure (4-Figure1_simulation.R) in palette and
# theme: viridis for the direct estimator (DNN w. Controls), bz encoded
# as colour position over the same domain c(-0.025, 4.25).
#
# Single method (DNN w. Controls = posthoc_xfit). Top row = direct
# image effect (fx); bottom row = nonlinear covariate effect (fz).
# Columns: MSPE, Bias², Var.  Bottom row uses a tighter, lower y-range
# since fz bias and variance are both small once the spline regression
# converges.

df <- read_csv("results/simulation_images/nonlinear_fz.csv") %>%
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

# Define the colour scale once and reuse the same R object across all
# panels, so patchwork's `plot_layout(guides = "collect")` recognises
# the identical guides and deduplicates them into a single colourbar.
# (Calling scale_color_viridis_c() inside make_panel() with TeX(...)
# creates a fresh expression object each time and breaks deduplication.)
BZ_COLOR_SCALE <- scale_color_viridis_c(
  begin = 0, end = 1, option = "viridis",
  limits = c(-0.025, 4.25), breaks = c(0, 1, 2, 3, 4),
  name = expression(beta[Z])
)

make_panel <- function(data, ylab,
                       ylim = c(0.4 * 1e-4, 1.25)) {
  ggplot(data, aes(x = n * 0.5, y = value, group = bz)) +
    geom_line(aes(color = bz), alpha = 0.8, linewidth = 0.8) +
    geom_point(aes(color = bz), alpha = 0.8, size = 0.8) +
    scale_x_log10(
      name = TeX("$N_{train}$ ($\\log_{10}$ scale)"),
      breaks = 100 * 2^(0:7)
    ) +
    scale_y_log10(name = ylab) +
    BZ_COLOR_SCALE +
    coord_cartesian(
      xlim = c(100, 100 * 2^7.05),
      ylim = ylim
    ) +
    shared_theme
}

# Shared y-range across both rows for direct visual comparison.
YLIM_FX <- c(1e-5, 1e-1)
YLIM_FZ <- c(1e-5, 1e-1)

# Top row — fx (direct image effect): MSPE, Bias², Var.
top_mspe <- make_panel(
  df %>% filter(effect == "fx", metric == "mspe"),
  ylab = TeX("$MSPE(\\hat{f}_X)$"),
  ylim = YLIM_FX
)
top_bias <- make_panel(
  df %>% filter(effect == "fx", metric == "bias2"),
  ylab = TeX("$Bias^2(\\hat{f}_X)$"),
  ylim = YLIM_FX
)
top_var <- make_panel(
  df %>% filter(effect == "fx", metric == "var"),
  ylab = TeX("$Var(\\hat{f}_X)$"),
  ylim = YLIM_FX
)

# Bottom row — fz (nonlinear covariate effect): MSPE, Bias², Var.
bot_mspe <- make_panel(
  df %>% filter(effect == "fz", metric == "mspe"),
  ylab = TeX("$MSPE(\\hat{f}_Z)$"),
  ylim = YLIM_FZ
)
bot_bias <- make_panel(
  df %>% filter(effect == "fz", metric == "bias2"),
  ylab = TeX("$Bias^2(\\hat{f}_Z)$"),
  ylim = YLIM_FZ
)
bot_var <- make_panel(
  df %>% filter(effect == "fz", metric == "var"),
  ylab = TeX("$Var(\\hat{f}_Z)$"),
  ylim = YLIM_FZ
)

# Show the colourbar INSIDE the top-left (MSPE(fx)) panel, in the
# bottom-left corner where the curves have already decayed below
# ~1e-3 (well below the colourbar's y-position). All other panels
# suppress their legends.
top_mspe <- top_mspe +
  theme(legend.position = c(0.32, 0.18),
        legend.direction = "horizontal",
        legend.key.width = unit(0.18, 'in'),
        legend.background = element_rect(color = NA, fill = NA))

top_bias <- top_bias + theme(legend.position = "none")
top_var  <- top_var  + theme(legend.position = "none")
bot_mspe <- bot_mspe + theme(legend.position = "none")
bot_bias <- bot_bias + theme(legend.position = "none")
bot_var  <- bot_var  + theme(legend.position = "none")

fig <- (top_mspe | top_bias | top_var) /
       (bot_mspe | bot_bias | bot_var)

ggsave("graphics/Fig_nonlinear_fz.pdf", fig,
       width = 6.0, height = 2.7, units = "in",
       dpi = 600, device = cairo_pdf)
