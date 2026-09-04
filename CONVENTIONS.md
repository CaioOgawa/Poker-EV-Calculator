# Convenções

## Style
- `ruff` para lint e format (configurado em `pyproject.toml`), line length 100.
- `mypy` com `disallow_untyped_defs` em `core/` e `engine/` (`make typecheck`, também no CI). `--strict` completo não vale a pena enquanto `treys` não tiver type stubs — `warn_return_any` sob `--strict` flagaria só o atrito de terceiros sem tipagem, não bugs reais.

## Naming
- Classes em `PascalCase`, métodos/funções em `snake_case`, constantes em `UPPER_SNAKE`.

## Cartas
- Notação `treys` em todo o projeto: rank + suit, ex. `'As'`, `'Td'`, `'2c'` (nunca `'10s'` ou `'AH'`).
- Board: lista de strings na mesma notação, 3-5 cartas.

## Dinheiro / EV
- Sempre em **big blinds** internamente (`engine/`, `risk/`, `simulation/`). Conversão para `$` só acontece na camada de output (`risk/bankroll/recommender.py` é a exceção documentada — ele converte banca em `$` pra BBs internamente porque precisa mapear pra stakes reais).
- Probabilidades e frequências: floats em `[0.0, 1.0]`, nunca percentuais (a conversão pra `%` é só na exibição — CLI, charts).

## Testes
- Cobertura mínima informal: 80%+ em `core/`, `engine/`, `risk/`. `vision/`/`output/` podem ser menores (dependem de peso treinado ou de `matplotlib`, respectivamente).
- Marcador `slow` (`pytest.mark.slow`) em testes de Monte Carlo/equity sweep que demoram — `make test-fast` roda sem eles, `make test` roda tudo.
- Testes que dependem de recurso opcional ausente (pesos YOLO treinados, `opencv`/`mss` não instalados) devem `skipif`, nunca falhar — ver `tests/unit/test_cli.py` (`requires_cv2`, `requires_trained_model`) pro padrão.
- Prefira invariantes verificáveis (soma de payoff = pot, RoR bate com Monte Carlo independente, etc.) a asserções contra "o que o código já devolve" — ver `docs/AUDITORIA-2026-08-26.md` pros exemplos.

## Commits
- [Conventional Commits](https://www.conventionalcommits.org/) (`feat:`, `fix:`, `chore:`, `test:`, `docs:`, `perf:`). A mensagem deve descrever o que mudou de fato, não "auto-commit" genérico.
- Sem trailers de atribuição de IA (`Co-Authored-By: Claude ...`, `Claude-Session: ...`) — decisão do mantenedor, reforçada em `~/.claude/CLAUDE.md` global.

## Docs locais vs. repositório
- `docs/` (ROADMAP, AUDITORIA, AVALIAÇÃO) é gitignored de propósito — são notas de sessão de trabalho, não documentação do projeto. `CONVENTIONS.md`/`CONTRIBUTING.md`/`README.md` na raiz é que são pra quem só clona o repo.
