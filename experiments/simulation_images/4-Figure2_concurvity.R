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

df_bz <- read_csv("results/simulation_images/concurvity.csv") %>%
  mutate(model = factor(
    model,
    levels = c("covar", "covar_conc_0.1", "covar_conc_1", "covar_conc_10",
               "ssn", "posthoc_web", "posthoc"),
    labels = c(
      "NAM",
      "NAM + Reg. (0.1)",
      "NAM + Reg. (1)",
      "NAM + Reg. (10)",
      "SSN",
      "Weber",
      "Pen. Refit")
  )) %>%
  filter(!is.na(model)) %>%
  mutate(effect = factor(effect, levels=c('y', 'fx', 'fr', 'fz')))

# Manual color scale:
#   NAM             = viridis purple   (no correction baseline)
#   NAM + Reg. (×3) = sequential blues (Siems concurvity penalty)
#   SSN             = red              (post-hoc orth on a NAM)
#   Weber           = orange           (post-hoc orth on a NN-only backbone)
#   Pen. Refit      = viridis yellow   (ours)
method_colors <- c(
  "NAM"              = "#440154",
  "NAM + Reg. (0.1)" = "#9ECAE1",
  "NAM + Reg. (1)"   = "#2171B5",
  "NAM + Reg. (10)"  = "#084594",
  "SSN"              = "#E31A1C",
  "Weber"            = "#FF7F00",
  "Pen. Refit"       = "#FDE725"
)



# Shared theme
shared_theme <- theme_bw() +
  theme(
    legend.title = element_text(vjust=-1.5),
    legend.title.position = "top",
    legend.key.height = unit(0.06, 'in'),
    legend.key.width = unit(0.15, 'in'),
    legend.ticks.length = unit(c(-.05, 0), 'in'),
    legend.ticks = element_line(color='black'),
    legend.position = c(0.44, 0.22),
    legend.direction = "vertical",
    legend.background = element_rect(color = NA, fill = NA),
    legend.text = element_text(size = 6, family = "serif", margin = margin(l = 2)),
    legend.margin = margin(0, 0, 0, 0),
    legend.spacing.x = unit(4, "pt"),
    axis.text.x = element_text(angle = 30, hjust = 1, vjust = 1.2, size=6),
    axis.text.y = element_text(hjust = 1.25, size=6),
    text = element_text(size = 8, family = "serif"),
    plot.margin = margin(2, 2, 2, 2)
  )

# Reusable plotting function
make_plot <- function(data, ylab, show_legend = TRUE, strip_labels = TRUE) {
  p <- ggplot(
    data,
    aes(x = n * 0.5, y = value, group = model)
  ) +
    geom_line(aes(color = model), alpha = 0.8, linewidth = 0.8) +
    geom_point(aes(color = model), alpha = 0.8, size = 0.8) +
    scale_x_log10(
      name = TeX("$N_{train}$ ($\\log_{10}$ scale)"),
      breaks = 100 * 2^(0:9)
    ) +
    scale_y_log10(name = ylab) +
    scale_color_manual(name = NULL, values = method_colors) +
    coord_cartesian(
      xlim = c(100, 100 * 2^9.5),
      ylim = c(10, .4 * 1e-4)
    ) +
    shared_theme
  
  if (!show_legend) {
    p <- p + theme(legend.position = "none")
  }
  if (strip_labels) {
    p <- p + theme(strip.text.y = element_blank())
  }
  
  p
}

# Build plots
METHODS <- c("NAM", "NAM + Reg. (0.1)", "NAM + Reg. (1)", "NAM + Reg. (10)",
             "SSN", "Weber", "Pen. Refit")

b1 <- make_plot(
  df_bz %>% filter(effect == "fx", metric == "bias2", model %in% METHODS),
  ylab = TeX("$Bias^2(\\hat{f}_X)$  ($\\log_{10}$ scale)"),
  show_legend = FALSE,
  strip_labels = FALSE
)

b2 <- make_plot(
  df_bz %>% filter(effect == "fx", metric == "var", model %in% METHODS),
  ylab = TeX("$Var(\\hat{f}_X)$  ($\\log_{10}$ scale)"),
  show_legend = TRUE,
  strip_labels = FALSE
)

b <- b1 + b2

ggsave("graphics/Fig2_Concurvity.pdf", b, width = 3.5, height = 1.8, units = "in", dpi=600, device = cairo_pdf)

