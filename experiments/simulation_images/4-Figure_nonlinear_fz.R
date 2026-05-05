pdf(NULL)  # suppress automatic Rplots.pdf when run non-interactively
library(readr)
library(dplyr)
library(ggplot2)
library(tikzDevice)
library(latex2exp)
library(patchwork)

# Mirrors the bias-bz figure (4-Figure1_simulation.R) in palette and
# theme: viridis for the direct estimator (DNN w. Controls), inferno for
# the orthogonalised estimator (DNN w. Controls + Orth.). bz is encoded
# as colour position over the same domain c(-0.025, 4.25).

df <- read_csv("results/simulation_images/nonlinear_fz.csv") %>%
  mutate(model = factor(
    model,
    levels = c("posthoc_xfit", "posthoc_orth_xfit"),
    labels = c("DNN w. Controls", "DNN w. Controls + Orth.")
  )) %>%
  filter(!is.na(model)) %>%
  mutate(effect = factor(effect, levels = c("y", "fx", "fr", "fz")))

# Shared theme — copied from 4-Figure1_simulation.R for visual parity.
shared_theme <- theme_bw() +
  theme(
    legend.title.position = "left",
    legend.key.height = unit(0.06, 'in'),
    legend.key.width = unit(0.15, 'in'),
    legend.ticks.length = unit(c(-.05, 0), 'in'),
    legend.ticks = element_line(color = 'black'),
    legend.position = c(0.5, 0.92),
    legend.direction = "horizontal",
    legend.margin = margin(0, unit = "inch"),
    legend.background = element_rect(color = NA, fill = NA),
    axis.text.x = element_text(angle = 30, hjust = 1, vjust = 1.2, size = 6),
    axis.text.y = element_text(hjust = 1.25, size = 6),
    axis.title.x = element_text(margin = margin(t = 0)),
    text = element_text(size = 8, family = "serif"),
    plot.margin = margin(2, 2, 2, 2)
  )

# Reusable plotting function — single method per panel, bz on viridis or
# inferno (matches Fig 1 convention for direct vs residual effects).
make_panel <- function(data, ylab, color_option,
                       show_legend = TRUE, ylim = c(0.4 * 1e-4, 1.25)) {
  p <- ggplot(data, aes(x = n * 0.5, y = value, group = bz)) +
    geom_line(aes(color = bz), alpha = 0.8, linewidth = 0.8) +
    geom_point(aes(color = bz), alpha = 0.8, size = 0.8) +
    scale_x_log10(
      name = TeX("$N_{train}$ ($\\log_{10}$ scale)"),
      breaks = 100 * 2^(0:7)
    ) +
    scale_y_log10(name = ylab) +
    scale_color_viridis_c(
      begin = 0, end = 1, option = color_option,
      limits = c(-0.025, 4.25), breaks = c(0, 1, 2, 3, 4),
      name = TeX("$\\beta_Z$")
    ) +
    coord_cartesian(
      xlim = c(100, 100 * 2^7.05),
      ylim = ylim
    ) +
    shared_theme

  if (!show_legend) p <- p + theme(legend.position = "none")
  p
}

# ── Top row: DNN w. Controls (viridis) — direct estimator. ────────────────────
top_fx <- make_panel(
  df %>% filter(model == "DNN w. Controls",
                effect == "fx", metric == "mspe"),
  ylab = TeX("$MSPE(\\hat{f}_X)$"),
  color_option = "viridis", show_legend = TRUE
)
top_fr <- make_panel(
  df %>% filter(model == "DNN w. Controls",
                effect == "fr", metric == "mspe"),
  ylab = TeX("$MSPE(\\hat{f}^{re}_X)$"),
  color_option = "viridis", show_legend = FALSE
)
top_fz_bias <- make_panel(
  df %>% filter(model == "DNN w. Controls",
                effect == "fz", metric == "bias2"),
  ylab = TeX("$Bias^2(\\hat{f}_Z)$"),
  color_option = "viridis", show_legend = FALSE
)
top_fz_var <- make_panel(
  df %>% filter(model == "DNN w. Controls",
                effect == "fz", metric == "var"),
  ylab = TeX("$Var(\\hat{f}_Z)$"),
  color_option = "viridis", show_legend = FALSE
)

# ── Bottom row: DNN w. Controls + Orth. (inferno) — residual estimator. ───────
bot_fx <- make_panel(
  df %>% filter(model == "DNN w. Controls + Orth.",
                effect == "fx", metric == "mspe"),
  ylab = TeX("$MSPE(\\hat{f}_X)$"),
  color_option = "inferno", show_legend = TRUE
)
bot_fr <- make_panel(
  df %>% filter(model == "DNN w. Controls + Orth.",
                effect == "fr", metric == "mspe"),
  ylab = TeX("$MSPE(\\hat{f}^{re}_X)$"),
  color_option = "inferno", show_legend = FALSE
)
bot_fz_bias <- make_panel(
  df %>% filter(model == "DNN w. Controls + Orth.",
                effect == "fz", metric == "bias2"),
  ylab = TeX("$Bias^2(\\hat{f}_Z)$"),
  color_option = "inferno", show_legend = FALSE
)
bot_fz_var <- make_panel(
  df %>% filter(model == "DNN w. Controls + Orth.",
                effect == "fz", metric == "var"),
  ylab = TeX("$Var(\\hat{f}_Z)$"),
  color_option = "inferno", show_legend = FALSE
)

# Compose: 2 rows × 4 cols.
fig <- (top_fx | top_fr | top_fz_bias | top_fz_var) /
       (bot_fx | bot_fr | bot_fz_bias | bot_fz_var)

ggsave("graphics/Fig_nonlinear_fz.pdf", fig,
       width = 7.16, height = 3.4, units = "in",
       dpi = 600, device = cairo_pdf)
