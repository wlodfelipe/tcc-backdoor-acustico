# Figuras da Seção 4, a partir de results/ (JSON e CSV gerados pelo experimento).
# Uso, a partir da raiz do repositório:  R_LIBS_USER=~/R/library Rscript R/figuras.R
# Saída: figures/*.pdf (vetorial, para o documento) e figures/*.png (300 dpi, para conferência).

suppressPackageStartupMessages({
  library(ggplot2); library(jsonlite); library(dplyr); library(tidyr); library(patchwork); library(scales)
})

FONTE <- "Liberation Serif"  # métricas da Times New Roman (ABNT)
SNRS <- c(40, 30, 25, 20, 15, 10, 0)
dir.create("figures", showWarnings = FALSE)

pct <- label_percent(accuracy = 1, decimal.mark = ",", suffix = "%")
num <- function(acc) label_number(accuracy = acc, decimal.mark = ",")
eixo_snr <- scale_x_reverse(breaks = SNRS, labels = paste(SNRS, "dB"))

tema <- theme_bw(base_size = 11, base_family = FONTE) +
  theme(panel.grid.minor = element_blank(), strip.background = element_blank(),
        strip.text = element_text(face = "bold"), legend.position = "bottom",
        plot.tag = element_text(face = "bold"))
theme_set(tema)
update_geom_defaults("text", list(family = FONTE))

salvar <- function(p, nome, largura = 16, altura = 10) {
  ggsave(file.path("figures", paste0(nome, ".pdf")), p, width = largura, height = altura,
         units = "cm", device = cairo_pdf)
  ggsave(file.path("figures", paste0(nome, ".png")), p, width = largura, height = altura,
         units = "cm", dpi = 300)
  cat("ok", nome, "\n")
}

rotulo_taxa <- function(r) ifelse(r == 0, "Referência", paste0(sub(".", ",", as.character(r * 100), fixed = TRUE), "%"))
NIVEIS_TAXA <- c("Referência", "0,5%", "1%", "2%", "5%")

ler_execucoes <- function(padrao) {
  arqs <- Sys.glob(padrao)
  bind_rows(lapply(arqs, function(f) {
    d <- fromJSON(f)
    tibble(run = d$run, rate = d$rate, seed = d$seed, snr = d$snr_db,
           triggered_asr = d$test$triggered_asr, clean_asr = d$test$clean_asr,
           acc = d$test$acc_clean, history = list(as_tibble(d$history)))
  }))
}

ic_t <- function(x) {  # intervalo t de 95% entre sementes, limitado a [0, 1]
  if (length(x) < 2) return(c(NA, NA))
  h <- qt(0.975, length(x) - 1) * sd(x) / sqrt(length(x))
  pmin(pmax(mean(x) + c(-h, h), 0), 1)
}

# ---------------------------------------------------------------------------------------------
# Figura 1 — compromisso entre eficácia e imperceptibilidade, por SNR (calibração, taxa de 5%, três sementes)
calib <- ler_execucoes("results/kaggle/tcc-d12-varredura-*/out/runs/rate5pct_snr*_seed*.json")
at <- fromJSON("results/analysis/ativacao_calibracao.json")
ef <- bind_rows(lapply(SNRS, function(s) {
  a <- at[[format(s)]]
  tibble(snr = s, `Ativação na SNR de envenenamento` = a[[format(s)]]$triggered_asr_mean,
         `Ativação a 10 dB` = a[["10"]]$triggered_asr_mean)
})) %>% pivot_longer(-snr, names_to = "ativ", values_to = "asr")

det <- bind_rows(lapply(SNRS, function(s) {
  dir <- if (s %in% c(25, 15)) "tcc-d12-deteccao-extra" else "tcc-deteccao-snr"
  d <- fromJSON(sprintf("results/kaggle/%s/out/phase1/s2_trigger_snr%g.json", dir, s))
  tibble(snr = s, ssim = d$all_clips$ssim_full$mean, mascarada = d$all_clips$masked_energy_frac$median)
}))

# validação cruzada (10 folds, semente 0): envenenamento a 5% e 30 dB, com IC por bootstrap por gravação de origem
vc <- fromJSON("results/analysis/validacao_cruzada.json")
v30 <- vc[["5% a 30 dB"]]
cv30 <- tibble(snr = 30,
  ativ = c("Ativação na SNR de envenenamento", "Ativação a 10 dB"),
  asr = c(v30$triggered_asr_pooled, v30$ativacao_10db_pooled),
  lo = c(v30$triggered_asr_pooled_boot[1], v30$ativacao_10db_pooled_boot[1]),
  hi = c(v30$triggered_asr_pooled_boot[2], v30$ativacao_10db_pooled_boot[2]))

pA <- ggplot(ef, aes(snr, asr, linetype = ativ, shape = ativ)) +
  geom_hline(yintercept = 0.85, linetype = "dashed", colour = "grey40") +
  geom_line() + geom_point(size = 2) +
  geom_errorbar(data = cv30, aes(x = snr - 1.2, ymin = lo, ymax = hi), width = 0.8, linetype = "solid",
                colour = "grey50", inherit.aes = FALSE, show.legend = FALSE) +
  geom_point(data = cv30, aes(x = snr - 1.2, y = asr, shape = ativ), colour = "grey50", size = 2,
             inherit.aes = FALSE, show.legend = FALSE) +
  annotate("text", x = 28.2, y = v30$ativacao_10db_pooled, label = "validação cruzada", hjust = 0,
           size = 2.8, colour = "grey35", family = FONTE) +
  eixo_snr + scale_y_continuous(labels = pct, limits = c(0, 1)) +
  labs(x = NULL, y = "Triggered ASR\n(teste, média)", linetype = NULL, shape = NULL, tag = "A") +
  theme(legend.position = "top")

pB <- ggplot(det, aes(snr, ssim)) +
  geom_hline(yintercept = 0.99, linetype = "dashed", colour = "grey40") +
  geom_line() + geom_point(size = 2) +
  eixo_snr + scale_y_continuous(labels = num(0.01)) +
  labs(x = NULL, y = "SSIM médio", tag = "B")

pC <- ggplot(det, aes(snr, mascarada)) +
  geom_line() + geom_point(size = 2) +
  eixo_snr + scale_y_continuous(labels = pct, limits = c(0, 1)) +
  labs(x = "SNR do gatilho de envenenamento", y = "Energia mascarada\n(mediana)", tag = "C") +
  theme(axis.title.x = element_text(margin = margin(t = 10)))

salvar(pA / pB / pC, "fig1_compromisso_snr", altura = 18)

# ---------------------------------------------------------------------------------------------
# Figura 2 — Triggered ASR por taxa de envenenamento na SNR selecionada (10 dB)
main <- read.csv("results/analysis/main_stage_runs.csv") %>%
  mutate(rotulo = factor(rotulo_taxa(rate), levels = NIVEIS_TAXA))
stopifnot(!anyNA(main$rotulo))
res2 <- main %>% group_by(rotulo) %>%
  summarise(media = mean(triggered_asr), lo = ic_t(triggered_asr)[1], hi = ic_t(triggered_asr)[2])

# a referência (0% em todas as sementes) fica na Tabela 3; o eixo começa em 70% para separar as sementes
envs <- filter(main, rate > 0) %>% mutate(rotulo = droplevels(rotulo))
res2 <- filter(res2, rotulo != "Referência") %>% mutate(rotulo = droplevels(rotulo))
set.seed(1)
p2 <- ggplot(envs, aes(rotulo, triggered_asr)) +
  geom_hline(yintercept = 0.85, linetype = "dashed", colour = "grey40") +
  geom_jitter(width = 0.07, height = 0, shape = 1, colour = "grey40", size = 2) +
  annotate("text", x = 4.95, y = 0.85, label = "critério da H1 (85%)", hjust = 1, vjust = 1.6, size = 3, family = FONTE) +
  geom_errorbar(data = res2, aes(y = media, ymin = lo, ymax = hi), width = 0.12) +
  geom_point(data = res2, aes(y = media), size = 2.8) +
  geom_text(data = res2, aes(y = media, label = label_percent(0.1, decimal.mark = ",")(media)),
            nudge_x = 0.2, hjust = 0, size = 3.2, family = FONTE) +
  scale_x_discrete(expand = expansion(add = c(0.5, 1.0))) +
  scale_y_continuous(labels = pct, limits = c(0.7, 1.0), breaks = seq(0.7, 1, 0.05)) +
  labs(x = "Taxa de envenenamento (SNR de 10 dB)", y = "Triggered ASR (teste)", tag = "A") +
  theme(panel.grid.major.x = element_blank())

# (B) validação cruzada: um ponto por fold (semente 0) e ASR agregada com IC por bootstrap por gravação de origem
vcf <- bind_rows(lapply(c("ref", "1%", "2%"), function(k) {
  tibble(rotulo = if (k == "ref") "Referência" else k, triggered_asr = vc[[k]]$per_fold$triggered_asr)
})) %>% mutate(rotulo = factor(rotulo, levels = c("Referência", "1%", "2%")))
vca <- bind_rows(lapply(c("ref", "1%", "2%"), function(k) {
  tibble(rotulo = if (k == "ref") "Referência" else k, media = vc[[k]]$triggered_asr_pooled,
         lo = vc[[k]]$triggered_asr_pooled_boot[1], hi = vc[[k]]$triggered_asr_pooled_boot[2])
})) %>% mutate(rotulo = factor(rotulo, levels = levels(vcf$rotulo)))
set.seed(1)
p2b <- ggplot(vcf, aes(rotulo, triggered_asr)) +
  geom_hline(yintercept = 0.85, linetype = "dashed", colour = "grey40") +
  geom_jitter(width = 0.07, height = 0, shape = 1, colour = "grey40", size = 2) +
  geom_errorbar(data = vca, aes(y = media, ymin = lo, ymax = hi), width = 0.12) +
  geom_point(data = vca, aes(y = media), size = 2.8) +
  geom_text(data = vca, aes(y = media, label = label_percent(0.1, decimal.mark = ",")(media)),
            nudge_x = 0.2, hjust = 0, size = 3.2, family = FONTE) +
  scale_x_discrete(expand = expansion(add = c(0.5, 0.8))) +
  scale_y_continuous(labels = pct, limits = c(0, 1), breaks = seq(0, 1, 0.2)) +
  labs(x = "Validação cruzada (SNR de 10 dB)", y = NULL, tag = "B") +
  theme(panel.grid.major.x = element_blank())
salvar(p2 + p2b + plot_layout(widths = c(4, 3)), "fig2_asr_por_taxa", altura = 8)

# ---------------------------------------------------------------------------------------------
# Figura 3 — ΔAcc por taxa, com intervalo de Welch (terceiro critério da H1)
ms <- fromJSON("results/analysis/main_stage.json")
dacc <- ms$per_rate %>% as_tibble() %>%
  mutate(rotulo = factor(rotulo_taxa(rate), levels = NIVEIS_TAXA[-1]),
         lo = sapply(dacc_pp_ci95_welch, `[`, 1), hi = sapply(dacc_pp_ci95_welch, `[`, 2))
p3 <- ggplot(dacc, aes(dacc_pp_mean, rotulo)) +
  geom_vline(xintercept = 0, colour = "grey60") +
  geom_vline(xintercept = 2, linetype = "dashed", colour = "grey40") +
  annotate("text", x = 2, y = Inf, label = "limite da H1 (2 p.p.)", hjust = 1.05, vjust = 1.5, size = 3) +
  geom_errorbar(aes(xmin = lo, xmax = hi), width = 0.15, orientation = "y") + geom_point(size = 2.5) +
  scale_x_continuous(labels = num(0.1)) +
  labs(x = expression(Delta * "Acc = Acc"[ref] - "Acc"[backdoor] ~ "(p.p.)"), y = "Taxa de envenenamento")
salvar(p3, "fig3_delta_acc", altura = 8)

# ---------------------------------------------------------------------------------------------
# Figura 4 — modelos treinados a 10 dB ativados por gatilhos de outras intensidades (análise adicional)
cross <- main %>% select(rotulo, starts_with("asr_trig")) %>%
  pivot_longer(-rotulo, names_to = "snr", values_to = "asr") %>%
  mutate(snr = as.numeric(sub("asr_trig", "", snr))) %>%
  group_by(rotulo, snr) %>% summarise(asr = mean(asr), .groups = "drop")
p4 <- ggplot(cross, aes(factor(snr, levels = SNRS, labels = paste(SNRS, "dB")), rotulo, fill = asr)) +
  geom_tile(colour = "white") +
  geom_text(aes(label = pct(asr), colour = asr > 0.6), size = 3.2, show.legend = FALSE) +
  scale_colour_manual(values = c(`TRUE` = "white", `FALSE` = "black")) +
  scale_fill_gradient(low = "grey95", high = "grey15", labels = pct, limits = c(0, 1), name = "ASR média") +
  labs(x = "SNR do gatilho aplicado na inferência", y = "Taxa de envenenamento\n(treino a 10 dB)") +
  theme(panel.grid = element_blank(), legend.position = "right")
salvar(p4, "fig4_ativacao_cruzada", altura = 8)

# ---------------------------------------------------------------------------------------------
# Figura 5 — Triggered ASR na validação a cada época (acompanhamento; varredura, semente 0)
hist <- calib %>% filter(seed == 0) %>% select(snr, history) %>% unnest(history)
p5 <- ggplot(hist, aes(epoch, val_triggered_asr, linetype = factor(snr, levels = SNRS, labels = paste(SNRS, "dB")),
                       shape = factor(snr, levels = SNRS, labels = paste(SNRS, "dB")))) +
  geom_line() + geom_point(size = 1.6) +
  scale_y_continuous(labels = pct, limits = c(0, 1)) + scale_x_continuous(breaks = 1:10) +
  labs(x = "Época", y = "Triggered ASR (validação)", linetype = "SNR", shape = "SNR")
salvar(p5, "fig5_asr_validacao_epocas", altura = 9)

# ---------------------------------------------------------------------------------------------
# Figura 6 — auditoria por FFT: limiar fixado nos folds 1-8 (limpas), detecção nos folds 9-10 sem street_music
fft <- fromJSON("results/analysis/origem_calibracao_fft.json")$fft_fora_da_amostra
obs <- function(f) fft[["10"]][[f]]$fpr_eval
rot <- c(`0.05` = sprintf("Limiar para 5%% de falsos positivos (observado: %s%%)", format(round(100 * obs("0.05") + 1e-9, 1), decimal.mark = ",", nsmall = 1)),
         `0.01` = sprintf("Limiar para 1%% de falsos positivos (observado: %s%%)", format(round(100 * obs("0.01") + 1e-9, 1), decimal.mark = ",", nsmall = 1)))
aud <- bind_rows(lapply(SNRS, function(s) bind_rows(lapply(names(rot), function(f)
  tibble(snr = s, fpr = rot[[f]], tpr = fft[[format(s)]][[f]]$tpr))))) %>%
  mutate(fpr = factor(fpr, levels = rot))
p6 <- ggplot(aud, aes(snr, tpr, linetype = fpr, shape = fpr)) +
  annotate("rect", xmin = 27.5, xmax = 42.5, ymin = -Inf, ymax = Inf, fill = "grey92") +
  annotate("text", x = 35, y = 0.97, label = "satisfazem o critério\nde similaridade (H2)", size = 3, family = FONTE, vjust = 1) +
  geom_vline(xintercept = 10, linetype = "dotted", colour = "grey40") +
  annotate("text", x = 10, y = 0.03, label = "SNR do ataque", size = 3, family = FONTE, hjust = -0.08) +
  geom_line() + geom_point(size = 2) +
  geom_text(data = filter(aud, snr %in% c(30, 10)), aes(label = label_percent(1)(tpr),
            vjust = ifelse(grepl("^Limiar para 1", fpr), 1.9, -0.9)), size = 3, family = FONTE, show.legend = FALSE) +
  eixo_snr + scale_y_continuous(labels = pct, limits = c(0, 1.05), breaks = seq(0, 1, 0.25)) +
  labs(x = "SNR do gatilho (mais forte à direita)", y = "Gravações com gatilho detectadas", linetype = NULL, shape = NULL) +
  guides(linetype = guide_legend(ncol = 1), shape = guide_legend(ncol = 1)) +
  theme(axis.title.x = element_text(margin = margin(t = 10)), panel.grid.major.x = element_blank())
salvar(p6, "fig6_auditoria_fft", altura = 10)
