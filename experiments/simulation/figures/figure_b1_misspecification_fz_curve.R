pdf(NULL)  # suppress automatic Rplots.pdf when run non-interactively
library(ggplot2)
library(tikzDevice)
library(latex2exp)

# Visualisation of the non-linear covariate effect used in the
# `nonlinear_fz` simulation block: f_z(Z) = bz * sin(2π (Z - 0.5)).
# A small companion plot for the appendix that motivates the figure
# right after it. Z is drawn uniformly on [0,1] in the simulation, so
# we plot the function at uniform samples (sorted) to mirror the
# observed data the model sees during training.

set.seed(0)
n_samples <- 400
z_samples <- sort(runif(n_samples, 0, 1))
bz_values <- c(0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4)
df <- expand.grid(z = z_samples, bz = bz_values)
df$fz <- df$bz * sin(2 * pi * (df$z - 0.5))

shared_theme <- theme_bw() +
  theme(
    legend.title.position = "top",
    legend.title = element_text(hjust = 0.5, margin = margin(b = 2)),
    legend.key.height = unit(0.06, 'in'),
    legend.key.width = unit(0.14, 'in'),
    legend.ticks.length = unit(c(-.05, 0), 'in'),
    legend.ticks = element_line(color = 'black'),
    legend.direction = "horizontal",
    legend.box.spacing = unit(0, 'pt'),
    legend.spacing.y = unit(0, 'pt'),
    legend.margin = margin(0, 0, 0, 0),
    legend.background = element_rect(color = NA, fill = NA),
    axis.text.x = element_text(size = 6),
    axis.text.y = element_text(size = 6),
    axis.title.x = element_text(margin = margin(t = 0)),
    text = element_text(size = 8, family = "serif"),
    plot.margin = margin(2, 2, 2, 2)
  )

p <- ggplot(df, aes(x = z, y = fz, group = bz, color = bz)) +
  geom_line(linewidth = 0.8, alpha = 0.85) +
  scale_color_viridis_c(
    begin = 0, end = 1, option = "viridis",
    limits = c(-0.025, 4.25), breaks = c(0, 1, 2, 3, 4),
    name = TeX("$\\beta_Z$")
  ) +
  scale_x_continuous(name = TeX("$z$"), breaks = c(0, 0.25, 0.5, 0.75, 1)) +
  scale_y_continuous(name = TeX("$f_Z(z) = \\beta_Z \\sin(2\\pi (z - 0.5))$"),
                     breaks = seq(-4, 4, by = 2)) +
  shared_theme +
  theme(legend.position = c(0.78, 0.20),
        legend.text = element_text(size = 5),
        legend.title = element_text(size = 6, hjust = 0.5,
                                    margin = margin(b = 1)))

ggsave("experiments/simulation/output/graphics/Fig_nonlinear_fz_curve.pdf", p,
       width = 3.0, height = 2.0, units = "in",
       dpi = 600, device = cairo_pdf)
