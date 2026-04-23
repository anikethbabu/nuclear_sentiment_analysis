# Public Opinion Now

## Interpretation

This is a live directional sample from accessible public sources, not a statistically representative poll.
It estimates current public/media stance from article titles/snippets and public discussion posts.

## Headline

- Items scored: 499
- Mean tone score: -0.302
- Predicted supportive share: 27.3%
- Predicted concerned share: 57.5%
- Predicted mixed/neutral share: 15.2%

## By Platform

| platform    |   items |   mean_tone_score |   median_tone_score |   supportive_share |   concerned_share |   neutral_share |   mean_confidence |
|:------------|--------:|------------------:|--------------------:|-------------------:|------------------:|----------------:|------------------:|
| ALL         |     499 |           -0.3021 |             -0.9232 |             0.2725 |            0.5752 |          0.1523 |            0.9206 |
| reddit      |     128 |           -0.3876 |             -0.9761 |             0.2734 |            0.6172 |          0.1094 |            0.9486 |
| hacker_news |     126 |           -0.1897 |             -0.9118 |             0.3651 |            0.5397 |          0.0952 |            0.9358 |
| google_news |     125 |           -0.1846 |             -0.8598 |             0.3520 |            0.5280 |          0.1200 |            0.9332 |
| gdelt       |     120 |           -0.4514 |             -0.9143 |             0.0917 |            0.6167 |          0.2917 |            0.8616 |

## By Topic Query

| query                 |   items |   mean_tone_score |   supportive_share |   concerned_share |
|:----------------------|--------:|------------------:|-------------------:|------------------:|
| nuclear energy        |     118 |           -0.1539 |             0.3644 |            0.4831 |
| small modular reactor |     110 |           -0.2057 |             0.3545 |            0.5818 |
| nuclear power         |      73 |           -0.2194 |             0.3288 |            0.5068 |
| nuclear reactor       |      89 |           -0.3782 |             0.2247 |            0.5843 |
| nuclear waste         |     109 |           -0.5530 |             0.0917 |            0.7064 |

## Collection Warnings

| platform   | query         | error                                                                                                                                                                                                                                                                                                                                                                                       |
|:-----------|:--------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| gdelt      | nuclear power | ConnectTimeout: HTTPSConnectionPool(host='api.gdeltproject.org', port=443): Max retries exceeded with url: /api/v2/doc/doc?query=%22nuclear+power%22&mode=ArtList&format=json&maxrecords=30&sort=HybridRel (Caused by ConnectTimeoutError(<HTTPSConnection(host='api.gdeltproject.org', port=443) at 0x273d98624e0>, 'Connection to api.gdeltproject.org timed out. (connect timeout=35)')) |

## X/Twitter Note

X/Twitter is not collected here because reliable search requires API access or credentials. Reddit and Hacker News are used as accessible public discussion proxies.