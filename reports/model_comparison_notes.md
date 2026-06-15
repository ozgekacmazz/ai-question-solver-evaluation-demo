# Model Comparison Benchmark Kullanım Notu

Bu benchmark, GPT, Gemini, Claude web arayüzleri ve projedeki yerel soru çözme tool'unu aynı 20 görsel soru üzerinde karşılaştırmak için hazırlanmıştır.

## Dosyalar

- Sorular: `data/comparison_questions/`
- Metadata: `data/comparison_questions/comparison_questions.json`
- Yerel tool sonuçları: `reports/comparison_tool_results.csv`
- Manuel web arayüzü şablonu: `reports/model_comparison_template.csv`

## Yerel Tool'u Çalıştırma

API anahtarı veya gerçek model bağlantısı yoksa mock/local mod ile çalıştır:

```bash
python scripts/run_comparison_benchmark.py --mock
```

Gerçek sağlayıcı ayarları aktifse mevcut pipeline modu ile çalıştır:

```bash
python scripts/run_comparison_benchmark.py --mode both
```

Alternatif modlar: `ocr`, `vision`, `both`, `adaptive`, `ocr_langflow`.

## Manuel Model Girişi

1. `reports/model_comparison_template.csv` dosyasını aç.
2. `data/comparison_questions/` altındaki PNG sorularını sırayla GPT, Gemini ve Claude web arayüzlerine yükle.
3. Modelin nihai şıkkını `gpt_answer`, `gemini_answer`, `claude_answer` kolonlarına yaz.
4. Doğruysa ilgili `*_correct` kolonunu `TRUE`, yanlışsa `FALSE` olarak işaretle.
5. `reports/comparison_tool_results.csv` dosyasındaki `tool_answer`, `tool_correct`, `ocr_text`, `ocr_issue` ve `error_type` değerlerini şablona aktar.

## Hata Ayrımı

- `ocr_issue`: OCR metni boş, eksik veya OCR adımında hata varsa doldurulur.
- `error_type=ocr_issue`: Yanlışlığın temel nedeni görüntü okuma/OCR tarafına işaret eder.
- `error_type=solution_or_model_error`: OCR metni var ama çözüm veya model cevabı yanlış görünür.
- `error_type=pipeline_error`: Pipeline beklenmeyen şekilde başarısız olmuştur.
- `error_type=no_answer`: Tool güvenilir bir cevap üretememiştir.

Bu ayrım, görsel okuma hatalarını modelin matematik/fizik çözüm hatalarından ayrı raporlamak için kullanılmalıdır.
