# Annotation Guidelines

How to assign grammar tags to split identifiers in [input/tagger_data.tsv](input/tagger_data.tsv). The tagset is listed in the [README](README.md#supported-tagset).

## Core principles

1. **Tag the word's role in the identifier, not its dictionary category.** A verb that modifies a noun is `NM` (`read Buffer` as a class name is `NM N`). A noun used as a function name's action is `V`.
2. **The last noun in a noun phrase is the head** (`N` or `NPL`). Every word before it that describes it is `NM`. `N N` in sequence should almost never happen.
3. **Tag by meaning, not by spelling.** Some words (`in`, `if`, `no`, `on`) have several meanings. Choose the meaning first, then the tag for that meaning (see [Ambiguous words](#ambiguous-words)).
4. **An abbreviation takes the tag of the word it stands for.** `if` meaning "interface" is `NM`, not `CJ`. `no` meaning "number" is `N`/`NM`, not `DT`.

## Deciding which meaning applies

Go through these in order and stop at the first one that settles it:

1. **TYPE.** A boolean usually means a state or a question ("is in X", "has no X"). A data type usually means a thing (input buffer, file number). If the type's name contains the word, the word is often an abbreviation of it (`on Factory` has type `ObjectNameFactory`, so `on` is "ObjectName").
2. **CONTEXT.** In `FUNCTION`, a leading word is usually an action. In `CLASS`, the words usually form one noun phrase.
3. **Sibling identifiers.** `a_low`, `a_cap` and `a_cost` next to `v_rhs` show that `a` means "arc", not a prefix.
4. **The source code** (GITHUB_URL). Only when 1–3 don't settle it.

If the source still doesn't settle it, choose the most likely tag and note the row as ambiguous. Don't invent a new convention for one row.

## Ambiguous words

| Word | Meaning → tag | Example |
|---|---|---|
| `in` | input, as a data variable or parameter → `NM` | `in buffer` NM N (`float`), `in channel` NM N |
| | membership or state, as a boolean or predicate → `P` | `in Literal` P N (`boolean`), `in best path` P NM N (`bool`) |
| | never `PRE`: it always carries meaning | |
| `out` | output → `NM` | `conv out channels` NM NM NPL |
| | part of "out of" → `P` | `out Of Memory` P P N |
| `for` | preposition → `P` | `build Schema For 2Dimensional Dataset` V N P NM N |
| | the `for` loop construct → `N` | `end for` V N, `parallel for` NM N |
| `on` | preposition, including event handlers → `P` | `commit on success` V P N, `on Click` P N |
| | abbreviation → the word it stands for | `on Factory` NM N (type `ObjectNameFactory`) |
| `as` | preposition or conversion → `P` | `as Array` P N, `as bounding box` P NM N |
| | abbreviation → the word it stands for | `as id` NM N (type `ASIdentifiers`) |
| `if` | conditional → `CJ` | `flag true if should convert` V N CJ V V |
| | "interface" → `NM` | `dp if index` PRE NM N |
| `no` | "none" or "not" → `DT` | `add no truncate cache` V DT V N |
| | "number" → `N` / `NM` | `file no` NM N, `no of edges` N P NPL |
| `of` | preposition → `P` | `end of record` N P N |
| | openFrameworks class prefix `OF` → `PRE` | `OF Android Window` PRE NM N |
| `first`, `last`, `next`, `all` | always `DT` | `last Write Index` DT NM N |
| `data` | mass noun → `N` as the head, `NM` as a modifier, never `NPL` | `accelerometer Data` NM N |

## Single letters and short prefixes

Tag a short leading token by what it contributes:

| It is… | Tag | Example |
|---|---|---|
| a naming-convention prefix that doesn't describe the head: member `m`, class `C`, Hungarian `b`/`p`/`f`/`g`/`s`, a library namespace | `PRE` | `m annotation` PRE N, `C Archive Loader` PRE NM N, `b Loop` PRE N, `png get row bytes` PRE V NM NPL |
| an abbreviation of a real word that modifies the head | `NM` | `a cap` NM N (`a` = arc) |
| the thing itself, i.e. the head | `N` | `a`, `m 11` N D (matrix element), `x` |

When the same letter appears across a system (`m` in every freeminer member, `vx` in every libvx identifier), tag it the same way everywhere.

## Open questions

These aren't settled by the current data. Decide each one once and add it to the tables above.

- **Trailing `out` / `in`:** `buf out` is currently NM N, but `bytes out` is NPL NM. Should a trailing direction word be the head (`N`) or a modifier placed after the noun (`NM`)?
- **Trailing modifiers in general:** `bytes Received` NPL NM and `Background Parser Private` NM N NM put `NM` after the head. Is that the intended convention for all words placed after the noun?
- **`for` with a following noun:** `for Annotation`, `for Attribute` and `for num` are currently NM N. Are they prepositions (`P N`) or abbreviations?
