pdf(NULL)  # suppress automatic Rplots.pdf when run non-interactively
library(readr)
library(dplyr)
library(ggplot2)
library(tikzDevice)
library(latex2exp)
library(patchwork)

# Concurvity vs backbone size. Two methods (NAM = end-to-end SGD,
# DNN with Controls = post-hoc xfit refit) sweep over (n, q). Curves
# are coloured by q on a viridis log gradient.

df <- read_csv("results/simulation_images/concurvity_q.csv") %>%
  filter(model %in% c("covar", "posthoc_xfit")) %>%
  mutate(model = factor(
    model,
    levels = c("covar", "posthoc_xfit"),
    labels = c("NAM", "DNN with Controls")
  )) %>%
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

# Single shared colour scale (log-q on viridis); reusing the same R
# object across panels so patchwork's guide collection deduplicates.
Q_COLOR_SCALE <- scale_color_viridis_c(
  option = "viridis", trans = "log",
  limits = c(2, 1024), breaks = c(2, 8, 32, 128, 512),
  name = TeX("$q$")
)

make_panel <- function(data, ylab, ylim = c(1e-4, 1.25)) {
  ggplot(data, aes(x = n * 0.5, y = value, group = q)) +
    geom_line(aes(color = q), alpha = 0.8, linewidth = 0.8) +
    geom_point(aes(color = q), alpha = 0.8, size = 0.8) +
    scale_x_log10(
      name = TeX("$N_{train}$ ($\\log_{10}$ scale)"),
      breaks = 100 * 2^(1:7)
    ) +
    scale_y_log10(name = ylab) +
    Q_COLOR_SCALE +
    coord_cartesian(
      xlim = c(175, 100 * 2^7.05),
      ylim = ylim
    ) +
    shared_theme +
    facet_grid(model ~ .)
}

# Top row — NAM. Bottom row — DNN with Controls.
nam_fx <- make_panel(
  df %>% filter(model == "NAM", effect == "fx", metric == "mspe"),
  ylab = TeX("$MSPE(\\hat{f}_X)$")
)
nam_fr <- make_panel(
  df %>% filter(model == "NAM", effect == "fr", metric == "mspe"),
  ylab = TeX("$MSPE(\\hat{f}^{re}_X)$")
)
ctrl_fx <- make_panel(
  df %>% filter(model == "DNN with Controls", effect == "fx", metric == "mspe"),
  ylab = TeX("$MSPE(\\hat{f}_X)$")
)
ctrl_fr <- make_panel(
  df %>% filter(model == "DNN with Controls", effect == "fr", metric == "mspe"),
  ylab = TeX("$MSPE(\\hat{f}^{re}_X)$")
)

# Place the q colourbar inside the top-left panel; suppress on others
# (matches Fig_nonlinear_fz convention).
nam_fx <- nam_fx +
  theme(legend.position = c(0.32, 0.93),
        legend.direction = "horizontal",
        legend.key.width = unit(0.15, 'in'),
        legend.background = element_rect(color = NA, fill = NA))
nam_fr  <- nam_fr  + theme(legend.position = "none")
ctrl_fx <- ctrl_fx + theme(legend.position = "none")
ctrl_fr <- ctrl_fr + theme(legend.position = "none")

fig <- (nam_fx | nam_fr) /
       (ctrl_fx | ctrl_fr)

ggsave("graphics/Fig_concurvity_q.pdf", fig,
       width = 4.16, height = 3.4, units = "in",
       dpi = 600, device = cairo_pdf)
