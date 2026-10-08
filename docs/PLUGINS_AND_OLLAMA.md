# Plugin Claude Code e Ollama in questo progetto

## Plugin: cosa tenere e quando
| Plugin | Uso nel progetto | Quando |
|---|---|---|
| superpowers | brainstorming, piano scritto, esecuzione con TDD, review | funzionalita' non banali (Fasi 1, 3, 4, 7) |
| mattpocock-skills | `diagnose` (debug a ipotesi e prove), `grill-with-docs`, `design-an-interface` | metriche sospette, bug, decisioni di design |
| code-simplifier | semplifica il codice appena scritto | a fine feature, prima del commit |
| ponytail | codice minimale | `lite` su script/esplorazione; **off** su preprocess, splits, metrics, llm |
| github | issue, branch, PR | una issue per fase |
| frontend-design | interfaccia | solo Fase 7 |
| cowork-plugin-management | nessuno | disattivalo |

Onesta': sono 7 plugin con istruzioni parzialmente in conflitto. Sovrapposizioni note:
- **TDD**: superpowers e mattpocock hanno ciascuno il proprio flusso. Usa quello di superpowers e
  di mattpocock solo `diagnose`/design.
- **ponytail contro superpowers**: uno spinge a codice minimo, l'altro a piani dettagliati. Se noti che il
  codice di valutazione diventa "troppo compatto", spegni ponytail.
- Ogni plugin attivo consuma contesto a ogni sessione. Se vedi risposte piu' lente o confuse, disattiva
  quelli che non servono alla fase corrente.
- Non c'e' garanzia che un plugin migliori il risultato: verifica sempre il diff e i test.

## Come attivare/disattivare
In una sessione interattiva: `/plugin`. Quel comando non e' disponibile in modalita' non interattiva
(`claude -p`), dove si usa la shell:
```
claude plugin list
claude plugin disable <nome-plugin>
claude plugin enable <nome-plugin>
```
Controlla `claude plugin --help` per le opzioni di scope (utente/progetto/locale) della tua versione:
non le ho verificate sulla tua installazione. In `.claude/settings.local.json.example` c'e' un
esempio di `enabledPlugins`; l'identificatore esatto (`nome@marketplace`) va letto da `claude plugin list`.
Le sessioni cloud non caricano i plugin installati in locale.

## Ollama: cosa fa e cosa NO
**Fa**: trasforma `candidate_facts(...)` (numeri gia' calcolati) in una scheda di 120 parole in italiano.
**Non fa**: classificare, calcolare, interpretare la curva. Ogni testo passa da `numbers_not_in_source`;
se Ollama non risponde si usa `template_report`.

Scelta del modello: la RTX 3050 ha 4-8 GB di VRAM a seconda della versione (controlla con
`nvidia-smi`). Modelli piccoli (circa 3-8B quantizzati) sono realistici; il training della CNN e Ollama
**non** vanno eseguiti insieme sulla stessa GPU, per via della VRAM.
Confronta 2 modelli nella Fase 6 su ~50 schede: quota di numeri sospetti, uso del fallback, tempo.
Un modello che inventa numeri nel 20% delle schede non va nella demo.

Uso rapido:
```
ollama list
export OLLAMA_MODEL=<uno dei modelli installati>
python scripts/vetting_report.py --data data/processed/synthetic.npz --model models/synthetic_cnn.pt --n 5
```
