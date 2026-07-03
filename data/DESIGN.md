# Dataset design notes

Status: reviewed, July 2026. Every pair was checked for naturalness, context force, congruence-label correctness, and EN-ZH parallelism; no items required changes.

## Task logic

Every row pairs a context and a target sentence with two single-sentence
continuations. The target sentence contains an expression that is ambiguous
or underspecified between two readings. The context is written to force one
of the readings. One continuation is congruent with the forced reading, the
other with the rival reading. The field `forced` records which continuation
slot (A or B) holds the congruent one. A model that resolves the
interpretation from context should assign higher probability to the
congruent continuation given context plus target.

The congruent continuation is placed in slot A for odd-numbered pairs and
in slot B for even-numbered pairs. Within every phenomenon by language cell
the forced label is therefore exactly 8 A and 8 B, so a scorer with a fixed
slot bias performs at chance. Reading types alternate in blocks of four
(pairs 1 to 4 and 9 to 12 carry one reading, pairs 5 to 8 and 13 to 16 the
other), so the forced slot is also balanced within each reading type.

## Phenomena

### scalar_implicature (si01 to si16)

English `some`, Mandarin 有些 or 一些. Pairs si01 to si08 force the
upper-bounded reading (some but not all) with episodic contexts in which the
speaker has exhaustively inspected the domain, so the choice of `some` over
`all` is maximally informative. Pairs si09 to si16 cancel the implicature:
`some` sits in a downward-entailing or sufficiency environment (conditional
antecedents, restrictors of rule statements), and the context states an
explicit lower-bound rule, so the at-least reading is forced.

### scope (sc01 to sc16)

English targets have the surface order `every ... a ...` and are genuinely
ambiguous. Half the contexts force surface scope (every > a, witnesses vary),
half force inverse scope (a > every, one shared witness). Mandarin surface
scope is rigid: a postverbal indefinite under 每...都 only has the
distributive-wide reading, and inverse scope in situ is unavailable. The
Mandarin items therefore express the forced reading with word order that is
natural in the language: surface-scope items use 每...都 with a postverbal
indefinite (每位成员都读了一本小说), and inverse-scope items front the
existential (有一首歌，每位选手都唱了). As a consequence the Mandarin scope
arm tests whether a model tracks overtly marked scope, while the English arm
tests context-driven disambiguation of a string that stays ambiguous. Direct
EN versus ZH comparison for this phenomenon must keep that asymmetry in mind.

### numeral (nu01 to nu16)

Bare `three` and Mandarin 三 plus classifier. Exact-forced contexts describe
exhaustive counting or registration (an inspector counts extinguishers one by
one), so `three` reports the full tally. At-least-forced contexts state an
explicit lower-bound rule (three signatures make a petition valid and more
are welcome), so the numeral is read as a satisfied threshold.

### distributivity (di01 to di16)

English targets are plural-subject sentences with an indefinite object and
contain neither `together` nor `each`, so the collective versus distributive
ambiguity lives in the target itself. Contexts force a collective reading
(one shared artefact, joint action) or a distributive one (individual
entries, sharing forbidden). Mandarin uses the overt marker: distributive
targets carry 都 (三个学徒都做了一个书架), collective targets are bare
(这三个学生搭了一个木筏). 都 marks the distributive reading overtly, so as
with scope the Mandarin arm is partly a test of marker tracking rather than
pure ambiguity resolution. This is the natural way to say these things in
Mandarin, and translationese was avoided on purpose.

## Parallelism and naturalness

EN and ZH rows of a pair share scenario, forced reading, and continuation
logic, but each language was written to be natural rather than literal.
Proper names are localised (Maria versus 小李). Classifiers, aspect marking,
and discourse particles follow ordinary usage. Continuations within a pair
are kept close in length (mean absolute difference about 2.5 characters)
because the scorer compares summed log probabilities.

## Known limitations

- Items are author-curated and have not yet had an independent native-speaker
  review pass in either language.
- The forced reading is only as strong as the context. Some contexts are
  entailing (the village has one hill), some are strongly biasing rather than
  strictly entailing.
- 16 pairs per phenomenon is small. The size was chosen so that every item
  could be hand-checked.
