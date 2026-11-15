# Figuras da Seção 4, a partir de results/ (JSON e CSV gerados pelo experimento).
# Uso, a partir da raiz do repositório:  R_LIBS_USER=~/R/library Rscript R/figuras.R
# Saída: figures/*.pdf (vetorial, para o documento) e figures/*.png (300 dpi, para conferência).

suppressPackageStartupMessages({
  library(ggplot2); library(jsonlite); library(dplyr); library(tidyr); library(patchwork); library(scales)
})

FONTE <- "Liberation Serif"  # métricas da Times New Roman (ABNT)
SNRS <- c(40, 30, 20, 10, 0)
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
# Figura 1 — compromisso entre eficácia e imperceptibilidade, por SNR (taxa de 5%)
calib <- ler_execucoes("results/kaggle/tcc-calibracao/out/runs/rate5pct_snr*_seed0.json") %>%
  bind_rows(ler_execucoes("results/kaggle/tcc-verificacao-snr*/out/runs/*.json"))
ef <- calib %>% group_by(snr) %>%
  summarise(media = mean(triggered_asr), n = n(), lo = ic_t(triggered_asr)[1], hi = ic_t(triggered_asr)[2])

det <- bind_rows(lapply(SNRS, function(s) {
  d <- fromJSON(sprintf("results/kaggle/tcc-deteccao-snr/out/phase1/s2_trigger_snr%g.json", s))
  tibble(snr = s, ssim = d$all_clips$ssim_full$mean, mascarada = d$all_clips$masked_energy_frac$median)
}))

pA <- ggplot(ef, aes(snr, media)) +
  geom_hline(yintercept = 0.85, linetype = "dashed", colour = "grey40") +
  annotate("text", x = 40, y = 0.85, label = "critério da H1 (85%)", vjust = -0.5, hjust = 0, size = 3) +
  geom_errorbar(aes(ymin = lo, ymax = hi), width = 1.2, na.rm = TRUE) +
  geom_line() + geom_point(size = 2) +
  geom_point(data = calib, aes(snr, triggered_asr), shape = 1, colour = "grey50", size = 1.5) +
  eixo_snr + scale_y_continuous(labels = pct, limits = c(0, 1)) +
  labs(x = NULL, y = "Triggered ASR (teste)", tag = "A")

pB <- ggplot(det, aes(snr, ssim)) +
  geom_hline(yintercept = 0.99, linetype = "dashed", colour = "grey40") +
  annotate("text", x = 0, y = 0.99, label = "critério da H2 (0,99)", vjust = -0.5, hjust = 1, size = 3) +
  geom_line() + geom_point(size = 2) +
  eixo_snr + scale_y_continuous(labels = num(0.01)) +
  labs(x = NULL, y = "SSIM médio", tag = "B")

pC <- ggplot(det, aes(snr, mascarada)) +
  geom_line() + geom_point(size = 2) +
  eixo_snr + scale_y_continuous(labels = pct, limits = c(0, 1)) +
  labs(x = "SNR do gatilho (gatilho mais forte à direita)", y = "Energia mascarada\n(mediana)", tag = "C")

salvar(pA / pB / pC, "fig1_compromisso_snr", altura = 18)

# ---------------------------------------------------------------------------------------------
# Figura 2 — Triggered ASR por taxa de envenenamento na SNR selecionada (10 dB)
main <- read.csv("results/analysis/main_stage_runs.csv") %>%
  mutate(rotulo = factor(rotulo_taxa(rate), levels = NIVEIS_TAXA))
stopifnot(!anyNA(main$rotulo))
res2 <- main %>% group_by(rotulo) %>%
  summarise(media = mean(triggered_asr), lo = ic_t(triggered_asr)[1], hi = ic_t(triggered_asr)[2])

p2 <- ggplot(main, aes(rotulo, triggered_asr)) +
  geom_hline(yintercept = 0.85, linetype = "dashed", colour = "grey40") +
  geom_jitter(width = 0.08, height = 0, shape = 1, colour = "grey45", size = 1.8) +
  geom_errorbar(data = res2, aes(y = media, ymin = lo, ymax = hi), width = 0.18) +
  geom_point(data = res2, aes(y = media), size = 2.5) +
  scale_y_continuous(labels = pct, limits = c(0, 1)) +
  labs(x = "Taxa de envenenamento (SNR de 10 dB)", y = "Triggered ASR (teste)")
salvar(p2, "fig2_asr_por_taxa")

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
# Figura 5 — Triggered ASR na validação a cada época (acompanhamento; calibração, semente 0)
hist <- calib %>% filter(seed == 0) %>% select(snr, history) %>% unnest(history)
p5 <- ggplot(hist, aes(epoch, val_triggered_asr, linetype = factor(snr, levels = SNRS, labels = paste(SNRS, "dB")),
                       shape = factor(snr, levels = SNRS, labels = paste(SNRS, "dB")))) +
  geom_line() + geom_point(size = 1.6) +
  scale_y_continuous(labels = pct, limits = c(0, 1)) + scale_x_continuous(breaks = 1:10) +
  labs(x = "Época", y = "Triggered ASR (validação)", linetype = "SNR", shape = "SNR")
salvar(p5, "fig5_asr_validacao_epocas", altura = 9)

# ---------------------------------------------------------------------------------------------
# Figura 6 — auditoria por FFT: taxa de detecção com falso positivo fixo em clipes limpos
fft <- fromJSON("results/analysis/fft_audit.json")
aud <- bind_rows(lapply(SNRS, function(s) {
  a <- fft[[format(s)]]$all_clips
  tibble(snr = s, `FPR de 1%` = a$fpr0.01$tpr, `FPR de 5%` = a$fpr0.05$tpr, auc = a$auc)
})) %>% pivot_longer(starts_with("FPR"), names_to = "fpr", values_to = "tpr")
p6 <- ggplot(aud, aes(snr, tpr, linetype = fpr, shape = fpr)) +
  geom_line() + geom_point(size = 2) +
  eixo_snr + scale_y_continuous(labels = pct, limits = c(0, 1)) +
  labs(x = "SNR do gatilho", y = "Taxa de detecção (TPR)", linetype = NULL, shape = NULL)
salvar(p6, "fig6_auditoria_fft", altura = 9)
