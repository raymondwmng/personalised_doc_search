# EUR-Lex Dataset Summary & Exploratory Analysis

**Source:** CEPS (*Centre for European Policy Studies*) EUR-Lex Dataset (`data/eurlex_kaggle`)  
**Time Range:** 1952 – August 2019  
**Total Acts:** 142,036 legal documents  
**Analysis Sample:** 2,900 documents sampled evenly across the dataset corpus  

---

## 1. Dataset Overview & File Inventory

The dataset comprises EU primary and secondary legislation scraped directly from the official [EUR-Lex](https://eur-lex.europa.eu) portal by CEPS researchers. It covers the legislative evolution of the European Union from its earliest treaties through August 2019.

| File Name | File Size | Document Count | Scope / Content |
| :--- | :--- | :--- | :--- |
| **`EurLex_all.csv`** | 1.58 GB | **142,036** | Complete master file with full English texts and metadata |
| **`EurLex_regulations_all.csv`** | 1.01 GB | **102,168** | Direct binding regulations (total) |
| ↳ `EurLex_regulations_1952_1990.csv` | 289 MB | 37,094 | Early community regulations (EEC / ECSC) |
| ↳ `EurLex_regulations_1990_2000.csv` | 248 MB | 32,653 | Post-Maastricht single market build-out |
| ↳ `EurLex_regulations_2000_2019.csv` | 481 MB | 32,423 | Modern EU regulations |
| **`EurLex_decisions.csv`** | 445 MB | **35,799** | Council & Commission decisions (addressed to specific states/firms) |
| **`EurLex_directives.csv`** | 119 MB | **4,071** | Framework directives (requiring national transposition) |
| **`CEPS_EurLex_codebook.pdf`** | 136 KB | — | Official metadata schema definition |

---

## 2. Metadata Schema & Variables

Each record contains up to 24 attributes:

| Variable Name | Present In | Description & Format | Example |
| :--- | :--- | :--- | :--- |
| `CELEX` | 142,035 | Unique CELEX identifier (`3` + Year + Type + ID) | `32019D0276` |
| `Act_name` | 139,889 | Full official legal title | `Decision (EU) 2019/276 of the European Parliament...` |
| `Act_type` | 142,035 | Legal instrument category | `Regulation`, `Directive`, `Decision`, `Delegated Regulation` |
| `Status` | 85,312 | Validity at scraping time | `In Force` vs `Not in Force` |
| `EUROVOC` | 91,613 | Multi-term controlled EuroVoc taxonomy keywords | `aid to refugees; budget appropriation; EC general budget` |
| `Subject_matter` | 91,613 | Broader thematic classification | `cooperation policy; budget; EU finance; migration` |
| `Treaty` | 109,648 | Treaty legal basis | `TFEU`, `TEEC`, `Euratom` |
| `Legal_basis_celex` | 109,231 | CELEX number of the enabling legal act | `32013Q1220(01)` |
| `Authors` | 142,035 | Enacting EU institution(s) | `European Parliament; European Council` |
| `Date_document` | 142,035 | Date of adoption (`YYYY-MM-DD`) | `2018-12-12` |
| `Date_publication` | 39,868 | Official Journal (OJ) publication date | `2019-02-22` |
| `First_entry_into_force`| 104,135 | Date of legal entry into force | `2019-01-01` |
| `Temporal_status` | 18,479 | Explicit sunset / expiration date | `2021-12-31` |
| `Act_cites` | 65,647 | Semicolon-separated CELEX IDs of cited legislation | `32013R1311; 32018D508` |
| `Act_amends` | 36,307 | CELEX of prior act modified or repealed | `32013R1311` |
| `Eurlex_link` | 142,036 | Web link to EUR-Lex legal notice | `https://eur-lex.europa.eu/...` |
| `ELI_link` | 142,035 | European Legislation Identifier URI | `http://data.europa.eu/eli/...` |
| `act_raw_text` | 134,633 | Raw English full text (recitals, legal articles, annexes) | Clean extracted text string |

---

## 3. Quantitative Characteristics (From 2,900 Sampled Acts)

### Act Types Distribution
- **Regulations:** 67.2% (`Regulation`: 1,950; `Implementing Regulation`: 137; `Delegated Regulation`: 13)
- **Decisions:** 25.8% (`Decision_ENTSCHEID`: 394; `Decision`: 288; `Decision_IMPL`: 36)
- **Directives:** 2.7% (`Directive`: 78; `Directive_IMPL`: 1)
  *(Note: While Directives make up a smaller proportion of act count, they are typically extensive framework laws that impact entire sectors across all 27 Member States).*

### Document Length Distribution
| Metric | Character Count | Word Count |
| :--- | :--- | :--- |
| **Minimum** | 42 | 3 |
| **25th Percentile** | 2,591 | 441 |
| **50th Percentile (Median)** | 3,835 | **681** |
| **75th Percentile** | 6,927 | 1,236 |
| **90th Percentile** | 17,908 | 2,985 |
| **95th Percentile** | 34,687 | 5,523 |
| **Maximum** | 1,019,131 | 164,366 |

### Metadata Coverage
- **Documents with EUROVOC:** 65.3% (~91,600 acts in full corpus)
- **Documents with Subject Matter:** 94.7% (~134,500 acts in full corpus)
- **Documents with Full Text:** 94.8% (~134,633 acts in full corpus)

### Most Frequent EUROVOC Concepts & Subjects
- **Top EUROVOC terms:** *Community aid to exports, automatic public tendering, import authorisation, entry price, origin of goods, third country, tariff quotas, cereals, rice, sugar, State aid, EC agreement*.
- **Top Subject Matters:** *trade policy, tariff policy, agricultural policy, cooperation policy, international trade, European construction, EU finance, health, technology and technical regulations*.

---

## 4. Searchable Domains & Topic Areas

The dataset can be effectively searched across eight major regulatory domains:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       Searchable Domains in EUR-Lex                         │
└──────────────────────────────────────┬──────────────────────────────────────┘
         ┌─────────────────────────────┼─────────────────────────────┐
         ▼                             ▼                             ▼
┌───────────────────────────┐ ┌───────────────────────────┐ ┌───────────────────────────┐
│ Agriculture & Fisheries   │ │ Trade, Customs & Tariffs  │ │ Internal Market & StateAid│
│ • Common Agri Policy (CAP)│ │ • Common Customs Tariff   │ │ • State Aid investigations│
│ • Commodity price regimes │ │ • Anti-dumping duties     │ │ • Antitrust & Cartels     │
│ • Total Allowable Catches │ │ • Rules of origin         │ │ • Public procurement     │
└───────────────────────────┘ └───────────────────────────┘ └───────────────────────────┘
         │                             │                             │
         ├─────────────────────────────┼─────────────────────────────┤
         ▼                             ▼                             ▼
┌───────────────────────────┐ ┌───────────────────────────┐ ┌───────────────────────────┐
│ Environment, Energy & ESG │ │ Finance, Tax & EU Budget  │ │ Justice, Borders & Asylum │
│ • LIFE programmes & Natura│ │ • Anti-Money Laundering   │ │ • Schengen acquis & SIS   │
│ • Emissions Trading (ETS) │ │ • Social Venture Capital  │ │ • Frontex & Border control│
│ • Hydrocarbon exploration │ │ • General EU budget       │ │ • European Arrest Warrant │
└───────────────────────────┘ └───────────────────────────┘ └───────────────────────────┘
         │                             │                             │
         └─────────────────────────────┼─────────────────────────────┘
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Transport, Digital & Public Health                                          │
│ • Aviation safety & SESAR airspace        • 5G spectrum allocation & telecom│
│ • Pharmaceuticals & EMA authorizations    • Consumer dispute resolution(ADR)│
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Representative Document Examples

### Example 1: Environment & Biodiversity Protection
- **CELEX:** `32018D0210`
- **Act Type:** Commission Implementing Decision
- **Date:** 2018-02-12
- **Title:** Commission Implementing Decision (EU) 2018/210 of 12 February 2018 on the adoption of the LIFE multiannual work programme for 2018-2020
- **EUROVOC:** `economic instrument for environmental protection`, `bio-economy`, `conservation of nature`, `Community framework programme`, `biodiversity`, `EU financing`
- **Subject Matter:** `environmental policy`, `economic policy`, `natural environment`, `EU finance`, `cooperation policy`
- **Snippet:** *"COMMISSION IMPLEMENTING DECISION (EU) 2018/210 of 12 February 2018 on the adoption of the LIFE multiannual work programme for 2018-2020... The LIFE Programme should support projects that contribute to the implementation of the EU Birds and Habitats Directives..."*

### Example 2: Competition & State Aid Control
- **CELEX:** `32016D2327`
- **Act Type:** Commission Decision
- **Date:** 2016-07-05
- **Title:** Commission Decision (EU) 2016/2327 of 5 July 2016 on State aid SA.19864 implemented by Belgium — Public financing of Brussels public IRIS hospitals
- **EUROVOC:** `public financing`, `Brussels region`, `clinic`, `Belgium`, `State aid`, `control of State aid`, `general-interest services`
- **Subject Matter:** `financing and investment`, `health`, `competition`, `consumption`, `economic policy`
- **Snippet:** *"COMMISSION DECISION (EU) 2016/2327 of 5 July 2016 on State aid SA.19864 implemented by Belgium — Public financing of Brussels public IRIS hospitals... considering whether public deficit funding constitutes compatible compensation for public service obligations..."*

### Example 3: Telecommunications & Digital Infrastructure
- **CELEX:** `32017D0899`
- **Act Type:** Decision of the European Parliament and of the Council
- **Date:** 2017-05-17
- **Title:** Decision (EU) 2017/899 on the use of the 470-790 MHz frequency band in the Union
- **EUROVOC:** `regulation of telecommunications`, `audio-visual communications`, `industrial innovation`, `radio telecommunications`, `compatibility`
- **Subject Matter:** `communications`, `research and intellectual property`, `technology and technical regulations`
- **Snippet:** *"DECISION (EU) 2017/899 OF THE EUROPEAN PARLIAMENT AND OF THE COUNCIL of 17 May 2017 on the use of the 470-790 MHz frequency band in the Union... Member States shall allow the use of the 694-790 MHz ('700 MHz') frequency band for terrestrial systems capable of providing wireless broadband electronic communications services..."*

### Example 4: Consumer Protection & eCommerce
- **CELEX:** `32016D1966`
- **Act Type:** Council Decision
- **Date:** 2016-09-20
- **Title:** Council Decision (EU) 2016/1966 concerning an amendment to Annex XIX (Consumer protection) to the EEA Agreement (Alternative Dispute Settlement System)
- **EUROVOC:** `Internet commerce`, `amendment of an agreement`, `ADR`, `consumer policy action plan`, `provision of services`, `EEA`
- **Subject Matter:** `marketing`, `justice`, `consumption`, `information and information processing`
- **Snippet:** *"COUNCIL DECISION (EU) 2016/1966 of 20 September 2016 on the position to be adopted... concerning an amendment to Annex XIX (Consumer protection) to the EEA Agreement... Directive 2013/11/EU of the European Parliament and of the Council on alternative dispute resolution for consumer disputes is to be incorporated..."*

### Example 5: Sustainable Fisheries & Bilateral Treaties
- **CELEX:** `32018D2068`
- **Act Type:** Council Decision
- **Date:** 2018-11-29
- **Title:** Council Decision (EU) 2018/2068 on the signing of the Sustainable Fisheries Partnership Agreement between the European Union and the Kingdom of Morocco
- **EUROVOC:** `EC agreement`, `signature of an agreement`, `Kingdom of Morocco`, `fishing area`, `fishing sustainability`, `protocol to an agreement`
- **Subject Matter:** `European construction`, `international affairs`, `Africa`, `fisheries`
- **Snippet:** *"COUNCIL DECISION (EU) 2018/2068 of 29 November 2018 on the signing, on behalf of the Union, of the Sustainable Fisheries Partnership Agreement between the European Union and the Kingdom of Morocco... promotes sustainable fishing in the fishing zone..."*

### Example 6: Financial Services & Social Venture Capital
- **CELEX:** `32019R0819`
- **Act Type:** Commission Delegated Regulation
- **Date:** 2019-02-01
- **Title:** Commission Delegated Regulation (EU) 2019/819 supplementing Regulation (EU) No 346/2013 with regard to conflicts of interest, social impact measurement and information to investors in the area of European social entrepreneurship funds
- **EUROVOC:** `disclosure of information`, `economic and social cohesion`, `investment policy`, `EC fund`, `conflict of interest`, `European social entrepreneurship fund`
- **Subject Matter:** `financing and investment`, `EU finance`, `labour law and working conditions`, `information technology`
- **Snippet:** *"COMMISSION DELEGATED REGULATION (EU) 2019/819 of 1 February 2019 supplementing Regulation (EU) No 346/2013 with regard to conflicts of interest, social impact measurement and information to investors in the area of European social entrepreneurship funds..."*

---

## 6. Recommendations for Retrieval & Personalization Architecture

1. **Chunking Strategy:**
   - 75% of legal acts are under 1,250 words; these can often be indexed in 2–3 coherent chunks (e.g. Title + Recitals, and Articles).
   - Long regulations (>10,000 words) should be sectioned hierarchically by Article / Chapter heading rather than arbitrary token boundaries.
2. **Metadata Filtering Facets:**
   - Expose explicit facets in the search engine for `Act_type` (Regulation vs Directive vs Decision), `Date_document` (e.g. decade selector), and `Status` (`In Force` vs `Not in Force`).
3. **EUROVOC & Subject Hierarchies:**
   - Incorporating `EUROVOC` keywords into the BM25 lexical index drastically boosts recall for domain terminology (e.g. searching *"5G"* matches `radio telecommunications` and `frequency band`).
4. **Citation Graph Utilization:**
   - The `Act_cites` and `Act_amends` fields enable PageRank / citation weighting to surface seminal, heavily cited baseline acts (such as foundational directives) ahead of minor procedural amendments.
