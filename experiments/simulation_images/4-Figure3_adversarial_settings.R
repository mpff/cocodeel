library(readr)
library(dplyr)
library(ggplot2)
library(tikzDevice)
library(latex2exp)
library(png)
library(grid)
library(patchwork)

# Load increasing fz dataset

effect_names <- c(
  'y' = 'y',
  'fx' = 'Image Effect',
  'fr' = 'Residual Image Effect',
  'fz' = 'Covariate Effect'
)

df_q <- read_csv("results/simulation_images/increasing_q.csv") %>%
  mutate(model = factor(
    model, 
    levels= c( "posthoc", "posthoc_orth", "base", "posthoc_web"),
    labels = c(
      "DNN w. Controls",
      "DNN w. Controls + Orth.",
      "DNN",
      "DNN + Regress Out [34]")
  )) %>%
  filter(n < 50000, q > 2) %>%
  mutate(effect = factor(effect, levels=c('y', 'fx', 'fr', 'fz')))

df_cv1 <- read_csv("results/simulation_images/increasing_cv.csv") %>%
  mutate(model = factor(
    model, 
    levels= c( "posthoc", "posthoc_orth", "base", "posthoc_web"),
    labels = c(
      "DNN w. Controls",
      "DNN w. Controls + Orth.",
      "DNN",
      "DNN + Regress Out [34]")
  )) %>%
  filter(n < 50000) %>%
  mutate(effect = factor(effect, levels=c('y', 'fx', 'fr', 'fz')))


# Shared theme
shared_theme <- theme_bw() +
  theme(
    legend.title.position = "left",
    legend.key.height = unit(0.06, 'in'),
    legend.key.width = unit(0.15, 'in'),
    legend.ticks.length = unit(c(-.05, 0), 'in'),
    legend.ticks = element_line(color='black'),
    legend.position = c(0.6, 0.85),
    legend.direction = "horizontal",
    legend.margin = margin(0, unit="inch"),
    legend.background = element_rect(color = NA, fill = NA),
    axis.text.x = element_text(angle = 30, hjust = 1, vjust = 1.2, size=6),
    axis.text.y = element_text(hjust = 1.25, size=6),
    text = element_text(size = 8, family = "serif"),
    plot.margin = margin(2, 2, 2, 2)   # reduce outer whitespace
  )

shared_theme2 <- theme_bw() +
  theme(
    legend.title.position = "left",
    legend.key.height = unit(0.06, 'in'),
    legend.key.width = unit(0.15, 'in'),
    legend.ticks.length = unit(c(-.05, 0), 'in'),
    legend.ticks = element_line(color='black'),
    legend.position = c(0.3, 0.15),
    legend.direction = "horizontal",
    legend.margin = margin(0, unit="inch"),
    legend.background = element_rect(color = NA, fill = NA),
    axis.text.x = element_text(angle = 30, hjust = 1, vjust = 1.2, size=6),
    axis.text.y = element_text(hjust = 1.25, size=6),
    text = element_text(size = 8, family = "serif"),
    plot.margin = margin(2, 2, 2, 2)   # reduce outer whitespace
  )



qDE <- ggplot(
  df_q %>% filter(
    effect == "fx",
    metric == "mspe",
    model %in% c("DNN w. Controls")
  ),
  aes(x = n * 0.5, y = value, group = q)
  ) +
  geom_line(aes(color = q), alpha = 0.8, linewidth = 0.8) +
  geom_point(aes(color = q), alpha = 0.8, size = 0.8) +
  scale_y_log10(name = TeX("$MSPE(\\hat{f}_X)$  ($\\log_{10}$ scale)")) +
  scale_x_log10(
      name = TeX("$N_{train}$ ($\\log_{10}$ scale)"),
      breaks = 100 * 2^(0:7)
    ) +
  scale_color_viridis_c(
      begin = 0, 
      end = 1, 
      option = "viridis",
      trans = "log",
      limits = c(3.85, 1050),
      breaks = c(4, 16, 64, 256, 1024)
    ) +
  labs(color = TeX("Number of features $q$")) +
  shared_theme +
  coord_cartesian(
      xlim = c(100, 100 * 2^7.5),
      ylim = c(0.15, 0.5*1e-3)
    ) +
  facet_grid(model ~ .)

qRE <- ggplot(
  df_q %>% filter(
    effect == "fr",
    metric == "mspe",
    model %in% c("DNN w. Controls + Orth.")
  ),
  aes(x = n * 0.5, y = value, group = q)
) +
  geom_line(aes(color = q), alpha = 0.8, linewidth = 0.8) +
  geom_point(aes(color = q), alpha = 0.8, size = 0.8) +
  scale_y_log10(name = TeX("$MSPE(\\hat{f}^{re}_X)$  ($\\log_{10}$ scale)")) +
  scale_x_log10(
    name = TeX("$N_{train}$ ($\\log_{10}$ scale)"),
    breaks = 100 * 2^(0:7)
  ) +
  scale_color_viridis_c(
    begin = 0, 
    end = 1, 
    option = "inferno",
    trans = "log",
    limits = c(3.85, 1050),
    breaks = c(4, 16, 64, 256, 1024)
  ) +
  labs(color = TeX("Number of features $q$")) +
  shared_theme +
  coord_cartesian(
    xlim = c(100, 100 * 2^7.5),
    ylim = c(0.15, 0.5*1e-3)
  ) +
  facet_grid(model ~ .)


qDE / qRE


cDE <- ggplot(
  df_cv1 %>% filter(
    effect == "fx",
    metric == "mspe",
    model %in% c("DNN w. Controls")
  ),
  aes(x = n * 0.5, y = value, group = cv1)
) +
  geom_line(aes(color = cv1), alpha = 0.8, linewidth = 0.8) +
  geom_point(aes(color = cv1), alpha = 0.8, size = 0.8) +
  scale_y_log10(name = TeX("$MSPE(\\hat{f}_X)$  ($\\log_{10}$ scale)")) +
  scale_x_log10(
    name = TeX("$N_{train}$ ($\\log_{10}$ scale)"),
    breaks = 100 * 2^(0:7)
  ) +
  scale_color_viridis_c(
    begin = 0, 
    end = 1, 
    option = "viridis"
  ) +
  labs(color = TeX("$corr(X,Z)$")) +
  shared_theme2 +
  coord_cartesian(
    xlim = c(100, 100 * 2^7.5),
    ylim = c(0.15, 0.5*1e-3)
  ) +
  facet_grid(model ~ .)

cRE <- ggplot(
  df_cv1 %>% filter(
    effect == "fr",
    metric == "mspe",
    model %in% c("DNN w. Controls + Orth.")
  ),
  aes(x = n * 0.5, y = value, group = cv1)
) +
  geom_line(aes(color = cv1), alpha = 0.8, linewidth = 0.8) +
  geom_point(aes(color = cv1), alpha = 0.8, size = 0.8) +
  scale_y_log10(name = TeX("$MSPE(\\hat{f}^{re}_X)$  ($\\log_{10}$ scale)")) +
  scale_x_log10(
    name = TeX("$N_{train}$ ($\\log_{10}$ scale)"),
    breaks = 100 * 2^(0:7)
  ) +
  scale_color_viridis_c(
    begin = 0, 
    end = 1, 
    option = "inferno"
  ) +
  labs(color = TeX("$corr(X,Z)$")) +
  shared_theme2 +
  coord_cartesian(
    xlim = c(100, 100 * 2^7.5),
    ylim = c(0.15, 0.5*1e-3)
  ) +
  facet_grid(model ~ .)


b <- (qDE + cDE) / (qRE + cRE)


ggsave("graphics/Fig4_simulation.pdf", b, width = 7.16, height = 3.8, units = "in", dpi=600, device = cairo_pdf)

