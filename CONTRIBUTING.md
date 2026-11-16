# Contributing

Detectability is maintained by one person, Mükremin Nergiz. Issues and pull requests are read and answered as time
allows; there is no guaranteed response time.

## Ways to help

- **Tell how it compared with a real survey.** Open an issue with the form
  ["I used Detectability on a real survey"](https://github.com/mukremin-nergiz/detectability/issues/new?template=used-on-a-survey.yml).
  Measured noise and a measured anomaly next to the modelled ones are the most useful contribution.
- **Report a result that looks wrong**, with the inputs and the source you compared against:
  ["A result looks wrong"](https://github.com/mukremin-nergiz/detectability/issues/new?template=result-looks-wrong.yml).
- **Suggest a source** for a material property, an instrument figure or a noise level. A figure enters the library only
  with a citable source that has been read.

## Changing the code

```
pip install -e ".[test]"
pytest
```

- Every physical formula carries its source (book or paper, with page or equation) in the code and has a test, either
  against a published value or against an independent derivation.
- Results are expectations under stated assumptions. A change that alters a number says so in `CHANGELOG.md`.
- Keep pull requests small and about one thing. British spelling in the documentation (modelled, licence).
- No telemetry, no network calls and no personal data in the library.

## Licence

Contributions are accepted under the project's BSD-3-Clause licence.
