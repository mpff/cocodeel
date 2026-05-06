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
      "DNN with Controls",
      "DNN with Controls\n+ Orthogonalisation",
      "DNN (Baseline)",
      "[17] DNN (Baseline)\n+ Orthogonalisation")
  )) %>%
  filter(n < 50000, q > 2) %>%
  mutate(effect = factor(effect, levels=c('y', 'fx', 'fr', 'fz')))

df_cv1 <- read_csv("results/simulation_images/increasing_cv.csv") %>%
  mutate(model = factor(
    model, 
    levels= c( "posthoc", "posthoc_orth", "base", "posthoc_web"),
    labels = c(
      "DNN with Controls",
      "DNN with Controls\n+ Orthogonalisation",
      "DNN (Baseline)",
      "[17] DNN (Baseline)\n+ Orthogonalisation")
  )) %>%
  filter(n < 50000) %>%
  mutate(effect = factor(effect, levels=c('y', 'fx', 'fr', 'fz')))

df_p <- read_csv("results/simulation_images/increasing_p.csv") %>%
  mutate(model = factor(
    model,
    levels= c( "posthoc", "posthoc_orth", "base", "posthoc_web"),
    labels = c(
      "DNN with Controls",
      "DNN with Controls\n+ Orthogonalisation",
      "DNN (Baseline)",
      "[17] DNN (Baseline)\n+ Orthogonalisation")
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
    axis.title.x = element_text(margin = margin(t = 0)),
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
    legend.position = c(0.33, 0.125),
    legend.direction = "horizontal",
    legend.margin = margin(0, unit="inch"),
    legend.background = element_rect(color = NA, fill = NA),
    axis.text.x = element_text(angle = 30, hjust = 1, vjust = 1.2, size=6),
    axis.text.y = element_text(hjust = 1.25, size=6),
    axis.title.x = element_text(margin = margin(t = 0)),
    text = element_text(size = 8, family = "serif"),
    plot.margin = margin(2, 2, 2, 2)   # reduce outer whitespace
  )



qDE <- ggplot(
  df_q %>% filter(
    effect == "fx",
    metric == "mspe",
    model %in% c("DNN with Controls")
  ),
  aes(x = n * 0.5, y = value, group = q)
  ) +
  geom_line(aes(color = q), alpha = 0.8, linewidth = 0.8) +
  geom_point(aes(color = q), alpha = 0.8, size = 0.8) +
  scale_y_log10(name = TeX("$MSPE(\\hat{f}_X)$")) +
  scale_x_log10(
      name = TeX("$N_{train}$ ($\\log_{10}$ scale)"),
      breaks = 100 * 2^(1:7)
    ) +
  scale_color_viridis_c(
      begin = 0, 
      end = 1, 
      option = "viridis",
      trans = "log",
      limits = c(3.85, 1050),
      breaks = c(4, 16, 64, 256, 1024)
    ) +
  labs(color = TeX("$q$")) +
  shared_theme +
  coord_cartesian(
      xlim = c(100, 100 * 2^7.5),
      ylim = c(0.5*1e-3, 0.15)
    ) +
  facet_grid(model ~ .)

qRE <- ggplot(
  df_q %>% filter(
    effect == "fr",
    metric == "mspe",
    model %in% c("DNN with Controls\n+ Orthogonalisation")
  ),
  aes(x = n * 0.5, y = value, group = q)
) +
  geom_line(aes(color = q), alpha = 0.8, linewidth = 0.8) +
  geom_point(aes(color = q), alpha = 0.8, size = 0.8) +
  scale_y_log10(name = TeX("$MSPE(\\hat{f}^{re}_X)$")) +
  scale_x_log10(
    name = TeX("$N_{train}$ ($\\log_{10}$ scale)"),
    breaks = 100 * 2^(1:7)
  ) +
  scale_color_viridis_c(
    begin = 0, 
    end = 1, 
    option = "inferno",
    trans = "log",
    limits = c(3.85, 1050),
    breaks = c(4, 16, 64, 256, 1024)
  ) +
  labs(color = TeX("$q$")) +
  shared_theme +
  coord_cartesian(
    xlim = c(100, 100 * 2^7.5),
    ylim = c(0.5*1e-3, 0.15)
  ) +
  facet_grid(model ~ .)


pDE <- ggplot(
  df_p %>% filter(
    effect == "fx",
    metric == "mspe",
    model %in% c("DNN with Controls")
  ),
  aes(x = n * 0.5, y = value, group = p)
) +
  geom_line(aes(color = p), alpha = 0.8, linewidth = 0.8) +
  geom_point(aes(color = p), alpha = 0.8, size = 0.8) +
  scale_y_log10(name = TeX("$MSPE(\\hat{f}_X)$")) +
  scale_x_log10(
    name = TeX("$N_{train}$ ($\\log_{10}$ scale)"),
    breaks = 100 * 2^(1:7)
  ) +
  scale_color_viridis_c(
    begin = 0, 
    end = 1, 
    option = "viridis",
    trans = "log",
    limits = c(0.95, 16.15),
    breaks = c(1, 2, 4, 8, 16)
  ) +
  labs(color = TeX("$p$")) +
  shared_theme +
  coord_cartesian(
    xlim = c(100, 100 * 2^7.5),
    ylim = c(0.5*1e-3, 0.15)
  ) +
  facet_grid(model ~ .)

pRE <- ggplot(
  df_p %>% filter(
    effect == "fr",
    metric == "mspe",
    model %in% c("DNN with Controls\n+ Orthogonalisation")
  ),
  aes(x = n * 0.5, y = value, group = p)
) +
  geom_line(aes(color = p), alpha = 0.8, linewidth = 0.8) +
  geom_point(aes(color = p), alpha = 0.8, size = 0.8) +
  scale_y_log10(name = TeX("$MSPE(\\hat{f}^{re}_X)$")) +
  scale_x_log10(
    name = TeX("$N_{train}$ ($\\log_{10}$ scale)"),
    breaks = 100 * 2^(1:7)
  ) +
  scale_color_viridis_c(
    begin = 0, 
    end = 1, 
    option = "inferno",
    trans = "log",
    limits = c(0.95, 16.15),
    breaks = c(1, 2, 4, 8, 16)
  ) +
  labs(color = TeX("$p$")) +
  shared_theme +
  coord_cartesian(
    xlim = c(100, 100 * 2^7.5),
    ylim = c(0.5*1e-3, 0.15)
  ) +
  facet_grid(model ~ .)



cDE <- ggplot(
  df_cv1 %>% filter(
    effect == "fx",
    metric == "mspe",
    model %in% c("DNN with Controls")
  ),
  aes(x = n * 0.5, y = value, group = cv1)
) +
  geom_line(aes(color = cv1), alpha = 0.8, linewidth = 0.8) +
  geom_point(aes(color = cv1), alpha = 0.8, size = 0.8) +
  scale_y_log10(name = TeX("$MSPE(\\hat{f}_X)$")) +
  scale_x_log10(
    name = TeX("$N_{train}$ ($\\log_{10}$ scale)"),
    breaks = 100 * 2^(1:7)
  ) +
  scale_color_viridis_c(
    begin = 0, 
    end = 1, 
    option = "viridis"
  ) +
  labs(color = TeX("$c$")) +
  shared_theme2 +
  coord_cartesian(
    xlim = c(100, 100 * 2^7.5),
    ylim = c( 0.5*1e-3, 0.15)
  ) +
  facet_grid(model ~ .)

cRE <- ggplot(
  df_cv1 %>% filter(
    effect == "fr",
    metric == "mspe",
    model %in% c("DNN with Controls\n+ Orthogonalisation")
  ),
  aes(x = n * 0.5, y = value, group = cv1)
) +
  geom_line(aes(color = cv1), alpha = 0.8, linewidth = 0.8) +
  geom_point(aes(color = cv1), alpha = 0.8, size = 0.8) +
  scale_y_log10(name = TeX("$MSPE(\\hat{f}^{re}_X)$")) +
  scale_x_log10(
    name = TeX("$N_{train}$ ($\\log_{10}$ scale)"),
    breaks = 100 * 2^(1:7)
  ) +
  scale_color_viridis_c(
    begin = 0, 
    end = 1, 
    option = "inferno"
  ) +
  labs(color = TeX("$c$")) +
  shared_theme2 +
  coord_cartesian(
    xlim = c(100, 100 * 2^7.5),
    ylim = c(0.5*1e-3, 0.15)
  ) +
  facet_grid(model ~ .)


b <- (qDE + pDE + cDE) / (qRE + pRE + cRE)
plot(b)


ggsave("graphics/Fig4_simulation.pdf", b, width = 7.16, height = 3.0, units = "in", dpi=600, device = cairo_pdf)

