pdf(NULL)  # suppress automatic Rplots.pdf when run non-interactively
library(readr)
library(dplyr)
library(ggplot2)
library(tikzDevice)
library(latex2exp)
library(patchwork)

# Concurvity vs backbone size. Two methods (DNN with Controls = post-hoc
# xfit refit, NAM = end-to-end SGD) sweep over (n, q). Top row = our
# method, bottom row = NAM baseline. Both rows share a single viridis
# colourbar over the params (in thousands), placed inside the top-left
# panel.

df <- read_csv("results/simulation_images/concurvity_q.csv") %>%
  filter(model %in% c("covar", "posthoc_xfit"),
         q >= 8) %>%   # drop the two smallest backbones (q=2,4 too small to fit fx)
  mutate(
    model = factor(
      model,
      levels = c("posthoc_xfit", "covar"),         # top row first
      labels = c("DNN with Controls", "NAM")
    ),
    params_k = (4801 + 514 * q) / 1000             # backbone params in thousands
  ) %>%
  mutate(effect = factor(effect, levels = c("y", "fx", "fr", "fz")))

shared_theme <- theme_bw() +
  theme(
    legend.key.height = unit(0.06, 'in'),
    legend.key.width = unit(0.18, 'in'),
    legend.ticks.length = unit(c(-.05, 0), 'in'),
    legend.ticks = element_line(color = 'black'),
    legend.direction = "horizontal",
    legend.title.position = "top",
    legend.title = element_text(hjust = 0.5, margin = margin(b = 2)),
    legend.box.spacing = unit(0, 'pt'),
    legend.spacing.y = unit(0, 'pt'),
    legend.margin = margin(0, 0, 0, 0),
    legend.background = element_rect(color = NA, fill = NA),
    axis.text.x = element_text(angle = 30, hjust = 1, vjust = 1.2, size = 6),
    axis.text.y = element_text(hjust = 1.25, size = 6),
    axis.title.x = element_text(margin = margin(t = 0)),
    text = element_text(size = 8, family = "serif"),
    plot.margin = margin(2, 2, 2, 2)
  )

# Single shared viridis colourbar, log-scale on params.
PARAMS_COLOR_SCALE <- scale_color_viridis_c(
  option = "viridis", trans = "log",
  limits = c(8.913, 531.137),
  breaks = c(8.913, 13.025, 21.249, 37.697, 70.593, 136.385, 267.969, 531.137),
  labels = c("9k", "13k", "21k", "38k", "71k", "136k", "268k", "531k"),
  name = "Parameters",
  guide = guide_colorbar(label.theme = element_text(
    size = 5, angle = 60, hjust = 1, vjust = 1, family = "serif"
  ))
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

# Top row — DNN with Controls.
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

# Bottom row — NAM.
nam_mspe <- make_panel(
  df %>% filter(model == "NAM", effect == "fx", metric == "mspe"),
  ylab = TeX("$MSPE(\\hat{f}_X)$")
)
nam_bias <- make_panel(
  df %>% filter(model == "NAM", effect == "fx", metric == "bias2"),
  ylab = TeX("$Bias^2(\\hat{f}_X)$")
)
nam_var  <- make_panel(
  df %>% filter(model == "NAM", effect == "fx", metric == "var"),
  ylab = TeX("$Var(\\hat{f}_X)$")
)

# Single colourbar at the bottom of the top-left (DNN/MSPE) panel.
ctrl_mspe <- ctrl_mspe +
  theme(legend.position = c(0.42, 0.18))
ctrl_bias <- ctrl_bias + theme(legend.position = "none")
ctrl_var  <- ctrl_var  + theme(legend.position = "none")
nam_mspe  <- nam_mspe  + theme(legend.position = "none")
nam_bias  <- nam_bias  + theme(legend.position = "none")
nam_var   <- nam_var   + theme(legend.position = "none")

fig <- (ctrl_mspe | ctrl_bias | ctrl_var) /
       (nam_mspe  | nam_bias  | nam_var)

ggsave("graphics/Fig_concurvity_q.pdf", fig,
       width = 6.0, height = 3.4, units = "in",
       dpi = 600, device = cairo_pdf)
