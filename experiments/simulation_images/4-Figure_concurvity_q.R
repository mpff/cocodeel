pdf(NULL)  # suppress automatic Rplots.pdf when run non-interactively
library(readr)
library(dplyr)
library(ggplot2)
library(tikzDevice)
library(latex2exp)
library(patchwork)

# Concurvity vs backbone size. Two methods (NAM = end-to-end SGD,
# DNN with Controls = post-hoc xfit refit) sweep over (n, q). Curves
# are coloured by the trainable parameter count of the backbone (in
# thousands), which is more interpretable than q itself for an ML
# audience. Param formula: BaseNetwork has 4801 + 514·q params.

df <- read_csv("results/simulation_images/concurvity_q.csv") %>%
  filter(model %in% c("covar", "posthoc_xfit"),
         q >= 8) %>%   # drop the two smallest backbones (q=2,4 too small to fit fx)
  mutate(
    model = factor(
      model,
      levels = c("covar", "posthoc_xfit"),
      labels = c("NAM", "DNN with Controls")
    ),
    params_k = (4801 + 514 * q) / 1000   # backbone params in thousands
  ) %>%
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

# Single shared colour scale (params_k on log-viridis); reusing the
# same R object across panels so patchwork's guide collection
# deduplicates. Breaks chosen at the actual Q_GRID positions
# corresponding to ~6k, 9k, 21k, 71k, 268k params (q = 2, 8, 32, 128,
# 512). Labels formatted with k suffix.
# `trans = "log"` makes colours interpolate exponentially in params.
# Breaks at q = 8, 32, 128, 512 (each 4× the previous) give clean
# geometric labels. Limits match the q >= 8 filter applied to the data.
PARAMS_COLOR_SCALE <- scale_color_viridis_c(
  option = "viridis", trans = "log",
  limits = c(8.913, 531.137),
  breaks = c(8.913, 21.249, 70.593, 267.969),
  labels = c("9k", "21k", "71k", "268k"),
  name = "Number of parameters"
)

make_panel <- function(data, ylab, ylim = c(1e-4, 1.25)) {
  ggplot(data, aes(x = n * 0.5, y = value, group = q)) +
    geom_line(aes(color = params_k), alpha = 0.8, linewidth = 0.8) +
    geom_point(aes(color = params_k), alpha = 0.8, size = 0.8) +
    scale_x_log10(
      name = TeX("$N_{train}$ ($\\log_{10}$ scale)"),
      breaks = 100 * 2^(1:7)
    ) +
    scale_y_log10(name = ylab) +
    PARAMS_COLOR_SCALE +
    coord_cartesian(
      xlim = c(175, 100 * 2^7.05),
      ylim = ylim
    ) +
    shared_theme +
    facet_grid(model ~ .)
}

# 2 rows (methods) × 3 cols (MSPE, Bias², Var) — all on fx.
nam_mspe  <- make_panel(
  df %>% filter(model == "NAM", effect == "fx", metric == "mspe"),
  ylab = TeX("$MSPE(\\hat{f}_X)$")
)
nam_bias  <- make_panel(
  df %>% filter(model == "NAM", effect == "fx", metric == "bias2"),
  ylab = TeX("$Bias^2(\\hat{f}_X)$")
)
nam_var   <- make_panel(
  df %>% filter(model == "NAM", effect == "fx", metric == "var"),
  ylab = TeX("$Var(\\hat{f}_X)$")
)
ctrl_mspe <- make_panel(
  df %>% filter(model == "DNN with Controls", effect == "fx", metric == "mspe"),
  ylab = TeX("$MSPE(\\hat{f}_X)$")
)
ctrl_bias <- make_panel(
  df %>% filter(model == "DNN with Controls", effect == "fx", metric == "bias2"),
  ylab = TeX("$Bias^2(\\hat{f}_X)$")
)
ctrl_var  <- make_panel(
  df %>% filter(model == "DNN with Controls", effect == "fx", metric == "var"),
  ylab = TeX("$Var(\\hat{f}_X)$")
)

# Single colourbar at the bottom of the top-left panel (NAM / MSPE(f̂_X))
# — the title "Number of parameters" sits above the colour bar via
# `legend.title.position = "top"`. Bottom-of-panel placement keeps it
# below the curves (which decay to ~3e-3 at large N, well above the
# legend at y=0.18). Suppress on all other panels.
nam_mspe <- nam_mspe +
  theme(legend.position = c(0.30, 0.18),
        legend.direction = "horizontal",
        legend.title.position = "top",
        legend.title = element_text(hjust = 0.5),
        legend.key.width = unit(0.12, 'in'),
        legend.background = element_rect(color = NA, fill = NA))
nam_bias  <- nam_bias  + theme(legend.position = "none")
nam_var   <- nam_var   + theme(legend.position = "none")
ctrl_mspe <- ctrl_mspe + theme(legend.position = "none")
ctrl_bias <- ctrl_bias + theme(legend.position = "none")
ctrl_var  <- ctrl_var  + theme(legend.position = "none")

fig <- (nam_mspe | nam_bias | nam_var) /
       (ctrl_mspe | ctrl_bias | ctrl_var)

ggsave("graphics/Fig_concurvity_q.pdf", fig,
       width = 6.0, height = 3.4, units = "in",
       dpi = 600, device = cairo_pdf)
