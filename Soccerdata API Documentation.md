## Get Started

```
    API Endpoint

        https://api.soccerdataapi.com
                
```

The Soccerdata API provides live scores, league stats and in-depth pre-match content for 125+ worldwide leagues.

Data types include live scores, league stats, transfers, injuries, head-to-head stats, white-label odds, A.I. powered match previews, projected and live team lineups, weather forecasts and game winner and over/under predictions.

To access the API endpoints, [sign-up for an account](https://soccerdataapi.com/) and obtain an **API key**.

An **API key** should be included in all requests using **auth\_token** as a parameter:

`api.soccerdataapi.com/livescores/?auth_token=320cae54d49a09f11c5cd23da43204a5543fb394`

JSON data is returned in gzip compressed format. Every API call must include the {'Accept-Encoding': 'gzip'} request header or it will fail. [More Info on Accept-Encoding headers.](https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Accept-Encoding)

## Get Country

```
                <code>
# Get Country: Javascript Example Request

<span>async</span> <span><span>function</span> <span>getCountries</span>(<span></span>) </span>{

    <span>const</span> response = <span>await</span> fetch(<span>"https://api.soccerdataapi.com/country/?auth_token=320cae54d49a09f11c5cd23da43204a5543fb394"</span>, {
        <span>method</span>: <span>'GET'</span>,
        <span>headers</span>: {
            <span>"Content-Type"</span>: <span>"application/json"</span>,
            <span>"Accept-Encoding"</span>: <span>"gzip"</span>
        },
    })
    .then(<span><span>response</span> =&gt;</span> {
        <span>return</span> response;
    })
    .catch(<span><span>error</span> =&gt;</span> {
        <span>return</span> error;
    });

    <span>const</span> data = <span>await</span> response.json();
    <span>console</span>.log(data);

}
                </code>
            
```

```
            <code>
<span># Get Country: Curl Example Request</span>
curl --request GET \
  --compressed \
  --header <span>'Content-Type: application/json'</span>--url <span>'https://api.soccerdataapi.com/country/?auth_token=320cae54d49a09f11c5cd23da43204a5543fb394'</span>
                </code>
            
```

```
            <code>
<span># Get Country: Python Example Request</span>
<span>import</span> requests

url = <span>"https://api.soccerdataapi.com/country/"</span>
querystring = {<span>'auth_token'</span>: <span>320</span>cae54d49a09f11c5cd23da43204a5543fb394}
headers = {
    <span>'Accept-Encoding'</span>: <span>'gzip'</span>,
    <span>'Content-Type'</span>: <span>'application/json'</span>
}
response = requests.get(url, headers=headers, params=querystring)
print(response.json())
                </code>
            
```

Retrieve a list of countries with a GET request to the endpoint:  
`https://api.soccerdataapi.com/country/`

  

```
            <code>
Get Country: Example JSON Response

{
    <span>"count"</span>: <span>221</span>,
    <span>"next"</span>: <span>null</span>,
    <span>"previous"</span>: <span>null</span>,
    <span>"results"</span>: [
        {
            <span>"id"</span>: <span>201</span>,
            <span>"name"</span>: <span>"afghanistan"</span>
        },
        {
            <span>"id"</span>: <span>47</span>,
            <span>"name"</span>: <span>"albania"</span>
        },
        {
            <span>"id"</span>: <span>87</span>,
            <span>"name"</span>: <span>"algeria"</span>
        },
        {
            <span>"id"</span>: <span>224</span>,
            <span>"name"</span>: <span>"american samoa"</span>
        },
        {
            <span>"id"</span>: <span>55</span>,
            <span>"name"</span>: <span>"andorra"</span>

        },

        ...
    ]
}
                </code>
            
```

## Get League

```
                <code>
# Get League: Javascript Example Request

<span>async</span> <span><span>function</span> <span>getLeagues</span>(<span></span>) </span>{

    <span>const</span> response = <span>await</span> fetch(<span>"https://api.soccerdataapi.com/league/?country_id=1&amp;auth_token=320cae54d49a09f11c5cd23da43204a5543fb394"</span>, {
        <span>method</span>: <span>'GET'</span>,
        <span>headers</span>: {
            <span>"Content-Type"</span>: <span>"application/json"</span>,
            <span>"Accept-Encoding"</span>: <span>"gzip"</span>
        },
    })
    .then(<span><span>response</span> =&gt;</span> {
        <span>return</span> response;
    })
    .catch(<span><span>error</span> =&gt;</span> {
        <span>return</span> error;
    });

    <span>const</span> data = <span>await</span> response.json();
    <span>console</span>.log(data);

}
                </code>
            
```

```
                <code>
<span># Get League: Curl Example Request</span>
curl --request GET \
  --compressed \
  --header <span>'Content-Type: application/json'</span>
  --url <span>'https://api.soccerdataapi.com/league/?country_id=1auth_token=320cae54d49a09f11c5cd23da43204a5543fb394'</span> 
                </code>
            
```

```
                <code>
<span># Get League: Python Example Request</span>
<span>import</span> requests

url = <span>"https://api.soccerdataapi.com/league/"</span>
querystring = {<span>'country_id'</span>: <span>1</span>, <span>'auth_token'</span>: <span>320</span>cae54d49a09f11c5cd23da43204a5543fb394}
headers = {
    <span>'Accept-Encoding'</span>: <span>'gzip'</span>,
    <span>'Content-Type'</span>: <span>'application/json'</span>
}
response = requests.get(url, headers=headers, params=querystring)
print(response.json())
                </code>
            
```

Retrieve a list of leagues with a GET request to the endpoint:  
`https://api.soccerdataapi.com/league/`

  

```
                <code>
Get League: Example JSON Response

{
    <span>"count"</span>: <span>129</span>,
    <span>"next"</span>: <span>null</span>,
    <span>"previous"</span>: <span>null</span>,
    <span>"results"</span>: [
        {
            <span>"id"</span>: <span>166</span>,
            <span>"country"</span>: {
                <span>"id"</span>: <span>1</span>,
                <span>"name"</span>: <span>"usa"</span>
            },
            <span>"name"</span>: <span>"USL Championship"</span>,
            <span>"is_cup"</span>: <span>false</span>
        },
        {
            <span>"id"</span>: <span>168</span>,
            <span>"country"</span>: {
                <span>"id"</span>: <span>1</span>,
                <span>"name"</span>: <span>"usa"</span>
            },
            <span>"name"</span>: <span>"MLS"</span>,
            <span>"is_cup"</span>: <span>false</span>
        },
        {
            <span>"id"</span>: <span>197</span>,
            <span>"country"</span>: {
                <span>"id"</span>: <span>2</span>,
                <span>"name"</span>: <span>"canada"</span>
            },
            <span>"name"</span>: <span>"Canadian Premier League"</span>,
            <span>"is_cup"</span>: <span>false</span>
        },
        {
            <span>"id"</span>: <span>198</span>,
            <span>"country"</span>: {
                <span>"id"</span>: <span>4</span>,
                <span>"name"</span>: <span>"europe"</span>
            },
            <span>"name"</span>: <span>"Europa Conference League"</span>,
            <span>"is_cup"</span>: <span>true</span>
        },

        ...
    ]
}
                </code>
            
```

#### QUERY PARAMETERS

| Field | Type | Description |
| --- | --- | --- |
| country\_id | Integer | (optional) Get leagues by country\_id |

## Get Season

```
                <code>
# Get Season: Javascript Example Request

<span>async</span> <span><span>function</span> <span>getSeasons</span>(<span></span>) </span>{

    <span>const</span> response = <span>await</span> fetch(<span>"https://api.soccerdataapi.com/season/?league_id=228&amp;auth_token=320cae54d49a09f11c5cd23da43204a5543fb394"</span>, {
        <span>method</span>: <span>'GET'</span>,
        <span>headers</span>: {
            <span>"Content-Type"</span>: <span>"application/json"</span>,
            <span>"Accept-Encoding"</span>: <span>"gzip"</span>
        },
    })
    .then(<span><span>response</span> =&gt;</span> {
        <span>return</span> response;
    })
    .catch(<span><span>error</span> =&gt;</span> {
        <span>return</span> error;
    });

    <span>const</span> data = <span>await</span> response.json();
    <span>console</span>.log(data);

}
                </code>
            
```

```
                <code>
<span># Get Season: Curl Example Request</span>
curl --request GET \
  --compressed \
  --header <span>'Content-Type: application/json'</span>--url <span>'https://api.soccerdataapi.com/season/?league_id=228&amp;auth_token=320cae54d49a09f11c5cd23da43204a5543fb394'</span> 
                </code>
            
```

```
                <code>
<span># Get Season: Python Example Request</span>
<span>import</span> requests

url = <span>"https://api.soccerdataapi.com/season/"</span>
querystring = {<span>'league_id'</span>: <span>228</span>, <span>'auth_token'</span>: <span>320</span>cae54d49a09f11c5cd23da43204a5543fb394}
headers = {
    <span>'Accept-Encoding'</span>: <span>'gzip'</span>,
    <span>'Content-Type'</span>: <span>'application/json'</span>
}
response = requests.get(url, headers=headers, params=querystring)
print(response.json())
                </code>
            
```

Retrieve a list of seasons for league with a GET request to the endpoint:  
`https://api.soccerdataapi.com/season/`

  

```
                <code>
Get Season: Example JSON Response

{
    <span>"count"</span>: <span>7</span>,
    <span>"next"</span>: <span>null</span>,
    <span>"previous"</span>: <span>null</span>,
    <span>"results"</span>: [
        {
            <span>"id"</span>: <span>4354</span>,
            <span>"league"</span>: {
                <span>"id"</span>: <span>228</span>,
                <span>"name"</span>: <span>"Premier League"</span>
            },
            <span>"year"</span>: <span>"2023-2024"</span>,
            <span>"is_active"</span>: <span>true</span>
        },
        {
            <span>"id"</span>: <span>3807</span>,
            <span>"league"</span>: {
                <span>"id"</span>: <span>228</span>,
                <span>"name"</span>: <span>"Premier League"</span>
            },
            <span>"year"</span>: <span>"2022-2023"</span>,
            <span>"is_active"</span>: <span>false</span>
        },
        {
            <span>"id"</span>: <span>3806</span>,
            <span>"league"</span>: {
                <span>"id"</span>: <span>228</span>,
                <span>"name"</span>: <span>"Premier League"</span>
            },
            <span>"year"</span>: <span>"2021-2022"</span>,
            <span>"is_active"</span>: <span>false</span>
        },
        {
            <span>"id"</span>: <span>3805</span>,
            <span>"league"</span>: {
                <span>"id"</span>: <span>228</span>,
                <span>"name"</span>: <span>"Premier League"</span>
            },
            <span>"year"</span>: <span>"2020-2021"</span>,
            <span>"is_active"</span>: <span>false</span>
        },
        {
            <span>"id"</span>: <span>3804</span>,
            <span>"league"</span>: {
                <span>"id"</span>: <span>228</span>,
                <span>"name"</span>: <span>"Premier League"</span>
            },
            <span>"year"</span>: <span>"2019-2020"</span>,
            <span>"is_active"</span>: <span>false</span>
        },
        {
            <span>"id"</span>: <span>3803</span>,
            <span>"league"</span>: {
                <span>"id"</span>: <span>228</span>,
                <span>"name"</span>: <span>"Premier League"</span>
            },
            <span>"year"</span>: <span>"2018-2019"</span>,
            <span>"is_active"</span>: <span>false</span>
        },
        {
            <span>"id"</span>: <span>3802</span>,
            <span>"league"</span>: {
                <span>"id"</span>: <span>228</span>,
                <span>"name"</span>: <span>"Premier League"</span>
            },
            <span>"year"</span>: <span>"2017-2018"</span>,
            <span>"is_active"</span>: <span>false</span>
        }
    ]
}

                </code>
            
```

#### QUERY PARAMETERS

| Field | Type | Description |
| --- | --- | --- |
| league\_id | Integer | (required) Get seasons by league\_id |

## Get Season Stages

```
                <code>
# Get Season Stages: Javascript Example Request

<span>async</span> <span><span>function</span> <span>getSeasonStages</span>(<span></span>) </span>{

    <span>const</span> response = <span>await</span> fetch(<span>"https://api.soccerdataapi.com/stage/?league_id=310&amp;season=2022-2023&amp;auth_token=320cae54d49a09f11c5cd23da43204a5543fb394"</span>, {
        <span>method</span>: <span>'GET'</span>,
        <span>headers</span>: {
            <span>"Content-Type"</span>: <span>"application/json"</span>,
            <span>"Accept-Encoding"</span>: <span>"gzip"</span>
        },
    })
    .then(<span><span>response</span> =&gt;</span> {
        <span>return</span> response;
    })
    .catch(<span><span>error</span> =&gt;</span> {
        <span>return</span> error;
    });

    <span>const</span> data = <span>await</span> response.json();
    <span>console</span>.log(data);

}
                </code>
            
```

```
                <code>
<span># Get Season Stages: Curl Example Request</span>
curl --request GET \
  --compressed \
  --header <span>'Content-Type: application/json'</span>--url <span>'https://api.soccerdataapi.com/stage/?league_id=310&amp;season=2022-2023&amp;auth_token=320cae54d49a09f11c5cd23da43204a5543fb394'</span>
                </code>
            
```

```
                <code>
<span># Get Season Stages: Python Example Request</span>
<span>import</span> requests

url = <span>"https://api.soccerdataapi.com/stage/"</span>
querystring = {<span>'league_id'</span>: <span>310</span>, <span>'season'</span>: <span>'2022-2023'</span>, <span>'auth_token'</span>: <span>320</span>cae54d49a09f11c5cd23da43204a5543fb394}
headers = {
    <span>'Accept-Encoding'</span>: <span>'gzip'</span>,
    <span>'Content-Type'</span>: <span>'application/json'</span>
}
response = requests.get(url, headers=headers, params=querystring)
print(response.json())
                </code>
            
```

Retrieve a list of stages for league season with a GET request to the endpoint:  
`https://api.soccerdataapi.com/stage/`

  

```
                <code>
Get Season Stage: Example JSON Response

{
    <span>"count"</span>: <span>11</span>,
    <span>"next"</span>: <span>null</span>,
    <span>"previous"</span>: <span>null</span>,
    <span>"results"</span>: [

        {
            <span>"id"</span>: <span>8667</span>,
            <span>"league"</span>: {
                <span>"id"</span>: <span>310</span>,
                <span>"name"</span>: <span>"UEFA Champions League"</span>
            },
            <span>"season"</span>: <span>"2022-2023"</span>,
            <span>"name"</span>: <span>"Preliminary Round - Semi-finals"</span>,
            <span>"has_groups"</span>: <span>false</span>,
            <span>"is_active"</span>: <span>false</span>
        },
        {
            <span>"id"</span>: <span>8666</span>,
            <span>"league"</span>: {
                <span>"id"</span>: <span>310</span>,
                <span>"name"</span>: <span>"UEFA Champions League"</span>
            },
            <span>"season"</span>: <span>"2022-2023"</span>,
            <span>"name"</span>: <span>"Preliminary Round - Final"</span>,
            <span>"has_groups"</span>: <span>false</span>,
            <span>"is_active"</span>: <span>false</span>
        },
        {
            <span>"id"</span>: <span>8662</span>,
            <span>"league"</span>: {
                <span>"id"</span>: <span>310</span>,
                <span>"name"</span>: <span>"UEFA Champions League"</span>
            },
            <span>"season"</span>: <span>"2022-2023"</span>,
            <span>"name"</span>: <span>"1st Qualifying Round"</span>,
            <span>"has_groups"</span>: <span>false</span>,
            <span>"is_active"</span>: <span>false</span>
        },
        {
            <span>"id"</span>: <span>8661</span>,
            <span>"league"</span>: {
                <span>"id"</span>: <span>310</span>,
                <span>"name"</span>: <span>"UEFA Champions League"</span>
            },
            <span>"season"</span>: <span>"2022-2023"</span>,
            <span>"name"</span>: <span>"2nd Qualifying Round"</span>,
            <span>"has_groups"</span>: <span>false</span>,
            <span>"is_active"</span>: <span>false</span>
        },
        {
            <span>"id"</span>: <span>8659</span>,
            <span>"league"</span>: {
                <span>"id"</span>: <span>310</span>,
                <span>"name"</span>: <span>"UEFA Champions League"</span>
            },
            <span>"season"</span>: <span>"2022-2023"</span>,
            <span>"name"</span>: <span>"3rd Qualifying Round"</span>,
            <span>"has_groups"</span>: <span>false</span>,
            <span>"is_active"</span>: <span>false</span>
        },
        {
            <span>"id"</span>: <span>8658</span>,
            <span>"league"</span>: {
                <span>"id"</span>: <span>310</span>,
                <span>"name"</span>: <span>"UEFA Champions League"</span>
            },
            <span>"season"</span>: <span>"2022-2023"</span>,
            <span>"name"</span>: <span>"Play-offs"</span>,
            <span>"has_groups"</span>: <span>false</span>,
            <span>"is_active"</span>: <span>false</span>
        },
        {
            <span>"id"</span>: <span>8646</span>,
            <span>"league"</span>: {
                <span>"id"</span>: <span>310</span>,
                <span>"name"</span>: <span>"UEFA Champions League"</span>
            },
            <span>"season"</span>: <span>"2022-2023"</span>,
            <span>"name"</span>: <span>"Group Stage"</span>,
            <span>"has_groups"</span>: <span>true</span>,
            <span>"is_active"</span>: <span>false</span>
        },
        {
            <span>"id"</span>: <span>8645</span>,
            <span>"league"</span>: {
                <span>"id"</span>: <span>310</span>,
                <span>"name"</span>: <span>"UEFA Champions League"</span>
            },
            <span>"season"</span>: <span>"2022-2023"</span>,
            <span>"name"</span>: <span>"Round of 16"</span>,
            <span>"has_groups"</span>: <span>false</span>,
            <span>"is_active"</span>: <span>false</span>
        },
        {
            <span>"id"</span>: <span>8644</span>,
            <span>"league"</span>: {
                <span>"id"</span>: <span>310</span>,
                <span>"name"</span>: <span>"UEFA Champions League"</span>
            },
            <span>"season"</span>: <span>"2022-2023"</span>,
            <span>"name"</span>: <span>"Quarter-finals"</span>,
            <span>"has_groups"</span>: <span>false</span>,
            <span>"is_active"</span>: <span>false</span>
        }
        {
            <span>"id"</span>: <span>8643</span>,
            <span>"league"</span>: {
                <span>"id"</span>: <span>310</span>,
                <span>"name"</span>: <span>"UEFA Champions League"</span>
            },
            <span>"season"</span>: <span>"2022-2023"</span>,
            <span>"name"</span>: <span>"Semi-finals"</span>,
            <span>"has_groups"</span>: <span>false</span>,
            <span>"is_active"</span>: <span>false</span>
        },
        {
            <span>"id"</span>: <span>8642</span>,
            <span>"league"</span>: {
                <span>"id"</span>: <span>310</span>,
                <span>"name"</span>: <span>"UEFA Champions League"</span>
            },
            <span>"season"</span>: <span>"2022-2023"</span>,
            <span>"name"</span>: <span>"Final"</span>,
            <span>"has_groups"</span>: <span>false</span>,
            <span>"is_active"</span>: <span>false</span>
        }
    ]
}

                </code>
            
```

#### QUERY PARAMETERS

| Field | Type | Description |
| --- | --- | --- |
| league\_id | Integer | (required) Get stages by league\_id. Defaults to current season. |
| season | String | (optional) Get stages by league\_id and season |

## Get Groups

```
                <code>
# Get Groups: Javascript Example Request

<span>async</span> <span><span>function</span> <span>getGroup</span>(<span></span>) </span>{

    <span>const</span> response = <span>await</span> fetch(<span>"https://api.soccerdataapi.com/group/?stage_id=8646&amp;auth_token=320cae54d49a09f11c5cd23da43204a5543fb394"</span>, {
        <span>method</span>: <span>'GET'</span>,
        <span>headers</span>: {
            <span>"Content-Type"</span>: <span>"application/json"</span>,
            <span>"Accept-Encoding"</span>: <span>"gzip"</span>
        },
    })
    .then(<span><span>response</span> =&gt;</span> {
        <span>return</span> response;
    })
    .catch(<span><span>error</span> =&gt;</span> {
        <span>return</span> error;
    });

    <span>const</span> data = <span>await</span> response.json();
    <span>console</span>.log(data);

}
                </code>
            
```

```
                <code>
<span># Get Groups: Curl Example Request</span>
curl --request GET \
  --compressed \
  --header <span>'Content-Type: application/json'</span>--url <span>'https://api.soccerdataapi.com/group/?stage_id=8646&amp;auth_token=320cae54d49a09f11c5cd23da43204a5543fb394'</span> 
                </code>
            
```

```
                <code>
<span># Get Groups: Python Example Request</span>
<span>import</span> requests

url = <span>"https://api.soccerdataapi.com/group/"</span>
querystring = {<span>'stage_id'</span>: <span>8646</span>, <span>'auth_token'</span>: <span>320</span>cae54d49a09f11c5cd23da43204a5543fb394}
headers = {
    <span>'Accept-Encoding'</span>: <span>'gzip'</span>,
    <span>'Content-Type'</span>: <span>'application/json'</span>
}
response = requests.get(url, headers=headers, params=querystring)
print(response.json())
                </code>
            
```

Retrieve a list of groups for stage with a GET request to the endpoint:  
`https://api.soccerdataapi.com/group/`

  

```
                <code>
Get Group: Example JSON Response

{
    <span>"count"</span>: <span>8</span>,
    <span>"next"</span>: <span>null</span>,
    <span>"previous"</span>: <span>null</span>,
    <span>"results"</span>: [
        {
            <span>"id"</span>: <span>689</span>,
            <span>"league"</span>: {
                <span>"id"</span>: <span>310</span>,
                <span>"name"</span>: <span>"UEFA Champions League"</span>
            },
            <span>"stage"</span>: {
                <span>"id"</span>: <span>8646</span>,
                <span>"name"</span>: <span>"Group Stage"</span>
            },
            <span>"name"</span>: <span>"Group A"</span>
        },
        {
            <span>"id"</span>: <span>690</span>,
            <span>"league"</span>: {
                <span>"id"</span>: <span>310</span>,
                <span>"name"</span>: <span>"UEFA Champions League"</span>
            },
            <span>"stage"</span>: {
                <span>"id"</span>: <span>8646</span>,
                <span>"name"</span>: <span>"Group Stage"</span>
            },
            <span>"name"</span>: <span>"Group B"</span>
        },
        {
            <span>"id"</span>: <span>691</span>,
            <span>"league"</span>: {
                <span>"id"</span>: <span>310</span>,
                <span>"name"</span>: <span>"UEFA Champions League"</span>
            },
            <span>"stage"</span>: {
                <span>"id"</span>: <span>8646</span>,
                <span>"name"</span>: <span>"Group Stage"</span>
            },
            <span>"name"</span>: <span>"Group C"</span>
        },
        {
            <span>"id"</span>: <span>692</span>,
            <span>"league"</span>: {
                <span>"id"</span>: <span>310</span>,
                <span>"name"</span>: <span>"UEFA Champions League"</span>
            },
            <span>"stage"</span>: {
                <span>"id"</span>: <span>8646</span>,
                <span>"name"</span>: <span>"Group Stage"</span>
            },
            <span>"name"</span>: <span>"Group D"</span>
        },
        {
            <span>"id"</span>: <span>693</span>,
            <span>"league"</span>: {
                <span>"id"</span>: <span>310</span>,
                <span>"name"</span>: <span>"UEFA Champions League"</span>
            },
            <span>"stage"</span>: {
                <span>"id"</span>: <span>8646</span>,
                <span>"name"</span>: <span>"Group Stage"</span>
            },
            <span>"name"</span>: <span>"Group E"</span>
        },
        {
            <span>"id"</span>: <span>694</span>,
            <span>"league"</span>: {
                <span>"id"</span>: <span>310</span>,
                <span>"name"</span>: <span>"UEFA Champions League"</span>
            },
            <span>"stage"</span>: {
                <span>"id"</span>: <span>8646</span>,
                <span>"name"</span>: <span>"Group Stage"</span>
            },
            <span>"name"</span>: <span>"Group F"</span>
        },
        {
            <span>"id"</span>: <span>695</span>,
            <span>"league"</span>: {
                <span>"id"</span>: <span>310</span>,
                <span>"name"</span>: <span>"UEFA Champions League"</span>
            },
            <span>"stage"</span>: {
                <span>"id"</span>: <span>8646</span>,
                <span>"name"</span>: <span>"Group Stage"</span>
            },
            <span>"name"</span>: <span>"Group G"</span>
        },
        {
            <span>"id"</span>: <span>696</span>,
            <span>"league"</span>: {
                <span>"id"</span>: <span>310</span>,
                <span>"name"</span>: <span>"UEFA Champions League"</span>
            },
            <span>"stage"</span>: {
                <span>"id"</span>: <span>8646</span>,
                <span>"name"</span>: <span>"Group Stage"</span>
            },
            <span>"name"</span>: <span>"Group H"</span>
        }
    ]
}

                </code>
            
```

#### QUERY PARAMETERS

| Field | Type | Description |
| --- | --- | --- |
| stage\_id | Integer | (required) Get groups by stage\_id |

## Get Stadium

```
                <code>
# Get Stadium: Javascript Example Request

<span>async</span> <span><span>function</span> <span>getStadium</span>(<span></span>) </span>{

    <span>const</span> response = <span>await</span> fetch(<span>"https://api.soccerdataapi.com/stadium/?team_id=4138&amp;auth_token=320cae54d49a09f11c5cd23da43204a5543fb394"</span>, {
        <span>method</span>: <span>'GET'</span>,
        <span>headers</span>: {
            <span>"Content-Type"</span>: <span>"application/json"</span>,
            <span>"Accept-Encoding"</span>: <span>"gzip"</span>
        },
    })
    .then(<span><span>response</span> =&gt;</span> {
        <span>return</span> response;
    })
    .catch(<span><span>error</span> =&gt;</span> {
        <span>return</span> error;
    });

    <span>const</span> data = <span>await</span> response.json();
    <span>console</span>.log(data);

}
                </code>
            
```

```
                <code>
<span># Get Stadium: Curl Example Request</span>
curl --request GET \
  --compressed \
  --header <span>'Content-Type: application/json'</span>--url <span>'https://api.soccerdataapi.com/stadium/?team_id=4138&amp;auth_token=320cae54d49a09f11c5cd23da43204a5543fb394'</span> 
                </code>
            
```

```
                <code>
<span># Get Stadium: Python Example Request</span>
<span>import</span> requests

url = <span>"https://api.soccerdataapi.com/stadium/"</span>
querystring = {<span>'team_id'</span>: <span>4138</span>, <span>'auth_token'</span>: <span>320</span>cae54d49a09f11c5cd23da43204a5543fb394}
headers = {
    <span>'Accept-Encoding'</span>: <span>'gzip'</span>,
    <span>'Content-Type'</span>: <span>'application/json'</span>
}
response = requests.get(url, headers=headers, params=querystring)
print(response.json())
                </code>
            
```

Retrieve stadium by team or stadium id with a GET request to the endpoint:  
`https://api.soccerdataapi.com/stadium/`  
Requires either team\_id or stadium\_id parameters.

  

```
                <code>
Get Stadium: Example JSON Response

{
    <span>"id"</span>: <span>2075</span>,
    <span>"teams"</span>: [
        {
            <span>"id"</span>: <span>4138</span>,
            <span>"name"</span>: <span>"Liverpool"</span>
        }
    ],
    <span>"name"</span>: <span>"Anfield"</span>,
    <span>"city"</span>: <span>"Liverpool"</span>
}

                </code>
            
```

#### QUERY PARAMETERS

| Field | Type | Description |
| --- | --- | --- |
| stadium\_id | Integer | (optionally required) Get stadium by stadium\_id |
| team\_id | Integer | (optionally required) Get stadium by team\_id |

## Get Team

```
                <code>
# Get Team: Javascript Example Request

<span>async</span> <span><span>function</span> <span>getTeam</span>(<span></span>) </span>{

    <span>const</span> response = <span>await</span> fetch(<span>"https://api.soccerdataapi.com/team/?team_id=4138&amp;auth_token=320cae54d49a09f11c5cd23da43204a5543fb394"</span>, {
        <span>method</span>: <span>'GET'</span>,
        <span>headers</span>: {
            <span>"Content-Type"</span>: <span>"application/json"</span>,
            <span>"Accept-Encoding"</span>: <span>"gzip"</span>
        },
    })
    .then(<span><span>response</span> =&gt;</span> {
        <span>return</span> response;
    })
    .catch(<span><span>error</span> =&gt;</span> {
        <span>return</span> error;
    });

    <span>const</span> data = <span>await</span> response.json();
    <span>console</span>.log(data);

}
                </code>
            
```

```
                <code>
<span># Get Team: Curl Example Request</span>
curl --request GET \
  --compressed \
  --header <span>'Content-Type: application/json'</span>--url <span>'https://api.soccerdataapi.com/team/?team_id=4138&amp;auth_token=320cae54d49a09f11c5cd23da43204a5543fb394'</span> 
                </code>
            
```

```
                <code>
<span># Get Team: Python Example Request</span>
<span>import</span> requests

url = <span>"https://api.soccerdataapi.com/team/"</span>
querystring = {<span>'team_id'</span>: <span>4138</span>, <span>'auth_token'</span>: <span>320</span>cae54d49a09f11c5cd23da43204a5543fb394}
headers = {
    <span>'Accept-Encoding'</span>: <span>'gzip'</span>,
    <span>'Content-Type'</span>: <span>'application/json'</span>
}
response = requests.get(url, headers=headers, params=querystring)
print(response.json())
                </code>
            
```

Retrieve team by id with a GET request to the endpoint:  
`https://api.soccerdataapi.com/team/`

  

```
                <code>
Get Team: Example JSON Response

{
    <span>"id"</span>: <span>4138</span>,
    <span>"name"</span>: <span>"Liverpool"</span>,
    <span>"country"</span>: {
        <span>"id"</span>: <span>8</span>,
        <span>"name"</span>: <span>"england"</span>
    },
    <span>"stadium"</span>: {
        <span>"id"</span>: <span>2075</span>,
        <span>"name"</span>: <span>"Anfield"</span>,
        <span>"city"</span>: <span>"Liverpool"</span>
    },
    <span>"is_nation"</span>: <span>false</span>
}

                </code>
            
```

#### QUERY PARAMETERS

| Field | Type | Description |
| --- | --- | --- |
| team\_id | Integer | (required) Get team by team\_id |

## Get Player

```
                <code>
# Get Player: Javascript Example Request

<span>async</span> <span><span>function</span> <span>getPlayer</span>(<span></span>) </span>{

    <span>const</span> response = <span>await</span> fetch(<span>"https://api.soccerdataapi.com/player/?player_id=61793&amp;auth_token=320cae54d49a09f11c5cd23da43204a5543fb394"</span>, {
        <span>method</span>: <span>'GET'</span>,
        <span>headers</span>: {
            <span>"Content-Type"</span>: <span>"application/json"</span>,
            <span>"Accept-Encoding"</span>: <span>"gzip"</span>
        },
    })
    .then(<span><span>response</span> =&gt;</span> {
        <span>return</span> response;
    })
    .catch(<span><span>error</span> =&gt;</span> {
        <span>return</span> error;
    });

    <span>const</span> data = <span>await</span> response.json();
    <span>console</span>.log(data);

}
                </code>
            
```

```
                <code>
<span># Get Player: Curl Example Request</span>
curl --request GET \
  --compressed \
  --header <span>'Content-Type: application/json'</span>--url <span>'https://api.soccerdataapi.com/player/?player_id=61793&amp;auth_token=320cae54d49a09f11c5cd23da43204a5543fb394'</span> 
                </code>
            
```

```
                <code>
<span># Get Player: Python Example Request</span>
<span>import</span> requests

url = <span>"https://api.soccerdataapi.com/player/"</span>
querystring = {<span>'player_id'</span>: <span>61793</span>, <span>'auth_token'</span>: <span>320</span>cae54d49a09f11c5cd23da43204a5543fb394}
headers = {
    <span>'Accept-Encoding'</span>: <span>'gzip'</span>,
    <span>'Content-Type'</span>: <span>'application/json'</span>
}
response = requests.get(url, headers=headers, params=querystring)
print(response.json())
                </code>
            
```

Retrieve player by id with a GET request to the endpoint:  
`https://api.soccerdataapi.com/player/`

  

```
                <code>
Get Player: Example JSON Response

{
    <span>"id"</span>: <span>61793</span>,
    <span>"name"</span>: <span>"J. Henderson"</span>,
    <span>"team"</span>: {
        <span>"id"</span>: <span>4138</span>,
        <span>"name"</span>: <span>"Liverpool"</span>
    }
}

                </code>
            
```

#### QUERY PARAMETERS

| Field | Type | Description |
| --- | --- | --- |
| player\_id | Integer | (required) Get player by player\_id |

## Get Transfers

```
                <code>
# Get Transfers: Javascript Example Request

<span>async</span> <span><span>function</span> <span>getTransfers</span>(<span></span>) </span>{

    <span>const</span> response = <span>await</span> fetch(<span>"https://api.soccerdataapi.com/transfers/?team_id=4138&amp;auth_token=320cae54d49a09f11c5cd23da43204a5543fb394"</span>, {
        <span>method</span>: <span>'GET'</span>,
        <span>headers</span>: {
            <span>"Content-Type"</span>: <span>"application/json"</span>,
            <span>"Accept-Encoding"</span>: <span>"gzip"</span>
        },
    })
    .then(<span><span>response</span> =&gt;</span> {
        <span>return</span> response;
    })
    .catch(<span><span>error</span> =&gt;</span> {
        <span>return</span> error;
    });

    <span>const</span> data = <span>await</span> response.json();
    <span>console</span>.log(data);

}
                </code>
            
```

```
                <code>
<span># Get Transfers: Curl Example Request</span>
curl --request GET \
  --compressed \
  --header <span>'Content-Type: application/json'</span>--url <span>'https://api.soccerdataapi.com/transfers/?team_id=4138&amp;auth_token=320cae54d49a09f11c5cd23da43204a5543fb394'</span> 
                </code>
            
```

```
                <code>
<span># Get Transfers: Python Example Request</span>
<span>import</span> requests

url = <span>"https://api.soccerdataapi.com/transfers/"</span>
querystring = {<span>'team_id'</span>: <span>4138</span>, <span>'auth_token'</span>: <span>320</span>cae54d49a09f11c5cd23da43204a5543fb394}
headers = {
    <span>'Accept-Encoding'</span>: <span>'gzip'</span>,
    <span>'Content-Type'</span>: <span>'application/json'</span>
}
response = requests.get(url, headers=headers, params=querystring)
print(response.json())
                </code>
            
```

Retrieve transfers by team\_id with a GET request to the endpoint:  
`https://api.soccerdataapi.com/transfers/`

  

```
                <code>
Get Transfers: Example JSON Response

{
    <span>"id"</span>: <span>4138</span>,
    <span>"name"</span>: <span>"Liverpool"</span>,
    <span>"transfers"</span>: {
        <span>"transfers_in"</span>: [
            {
                <span>"player_id"</span>: <span>27537</span>,
                <span>"player_name"</span>: <span>"L. Clarkson"</span>,
                <span>"from_team"</span>: {
                    <span>"id"</span>: <span>2717</span>,
                    <span>"name"</span>: <span>"Aberdeen"</span>
                },
                <span>"transfer_date"</span>: <span>"14-06-2023"</span>,
                <span>"transfer_type"</span>: <span>"n/a"</span>,
                <span>"transfer_amount"</span>: <span>0</span>,
                <span>"transfer_currency"</span>: <span>"usd"</span>
            },
            {
                <span>"player_id"</span>: <span>61790</span>,
                <span>"player_name"</span>: <span>"A. Mac Allister"</span>,
                <span>"from_team"</span>: {
                    <span>"id"</span>: <span>3200</span>,
                    <span>"name"</span>: <span>"Brighton &amp; Hove Albion"</span>
                },
                <span>"transfer_date"</span>: <span>"14-06-2023"</span>,
                <span>"transfer_type"</span>: <span>"transfer-fee"</span>,
                <span>"transfer_amount"</span>: <span>42000000</span>,
                <span>"transfer_currency"</span>: <span>"eur"</span>
            },

            ...

        ],
        <span>"transfers_out"</span>: [
            {
                <span>"player_id"</span>: <span>27486</span>,
                <span>"player_name"</span>: <span>"R. Williams"</span>,
                <span>"to_team"</span>: {
                    <span>"id"</span>: <span>2717</span>,
                    <span>"name"</span>: <span>"Aberdeen"</span>
                },
                <span>"transfer_date"</span>: <span>"28-06-2023"</span>,
                <span>"transfer_type"</span>: <span>"loan"</span>,
                <span>"transfer_amount"</span>: <span>0</span>,
                <span>"transfer_currency"</span>: <span>"usd"</span>
            },
            {
                <span>"player_id"</span>: <span>27537</span>,
                <span>"player_name"</span>: <span>"L. Clarkson"</span>,
                <span>"to_team"</span>: {
                    <span>"id"</span>: <span>2717</span>,
                    <span>"name"</span>: <span>"Aberdeen"</span>
                },
                <span>"transfer_date"</span>: <span>"15-06-2023"</span>,
                <span>"transfer_type"</span>: <span>"n/a"</span>,
                <span>"transfer_amount"</span>: <span>0</span>,
                <span>"transfer_currency"</span>: <span>"usd"</span>
            },

            ...

        ]
    }
}

                </code>
            
```

#### QUERY PARAMETERS

| Field | Type | Description |
| --- | --- | --- |
| team\_id | Integer | (required) Get transfers by team\_id |

## Get Head To Head

```
                <code>
# Get Head To Head: Javascript Example Request

<span>async</span> <span><span>function</span> <span>getHeadToHead</span>(<span></span>) </span>{

    <span>const</span> response = <span>await</span> fetch(<span>"https://api.soccerdataapi.com/head-to-head/?team_1_id=4137&amp;team_2_id=4149&amp;auth_token=320cae54d49a09f11c5cd23da43204a5543fb394"</span>, {
        <span>method</span>: <span>'GET'</span>,
        <span>headers</span>: {
            <span>"Content-Type"</span>: <span>"application/json"</span>,
            <span>"Accept-Encoding"</span>: <span>"gzip"</span>
        },
    })
    .then(<span><span>response</span> =&gt;</span> {
        <span>return</span> response;
    })
    .catch(<span><span>error</span> =&gt;</span> {
        <span>return</span> error;
    });

    <span>const</span> data = <span>await</span> response.json();
    <span>console</span>.log(data);

}
                </code>
            
```

```
                <code>
<span># Get Head To Head: Curl Example Request</span>
curl --request GET \
  --compressed \
  --header <span>'Content-Type: application/json'</span>--url <span>'https://api.soccerdataapi.com/head-to-head/?team_1_id=4137&amp;team_2_id=4149&amp;auth_token=320cae54d49a09f11c5cd23da43204a5543fb394'</span> 
                </code>
            
```

```
                <code>
<span># Get Head To Head: Python Example Request</span>
<span>import</span> requests

url = <span>"https://api.soccerdataapi.com/head-to-head/"</span>
querystring = {<span>'team_1_id'</span>: <span>4137</span>, <span>'team_2_id'</span>: <span>4149</span>, <span>'auth_token'</span>: <span>320</span>cae54d49a09f11c5cd23da43204a5543fb394}
headers = {
    <span>'Accept-Encoding'</span>: <span>'gzip'</span>,
    <span>'Content-Type'</span>: <span>'application/json'</span>
}
response = requests.get(url, headers=headers, params=querystring)
print(response.json())
                </code>
            
```

Retrieve head to head stats by team\_ids with a GET request to the endpoint:  
`https://api.soccerdataapi.com/head-to-head/`

  

```
                <code>
Get Head To Head: Example JSON Response

{
    <span>"id"</span>: <span>2893</span>,
    <span>"team1"</span>: {
        <span>"id"</span>: <span>4137</span>,
        <span>"name"</span>: <span>"Manchester United"</span>
    },
    <span>"team2"</span>: {
        <span>"id"</span>: <span>4149</span>,
        <span>"name"</span>: <span>"Nottingham Forest"</span>
    },
    <span>"stats"</span>: {
        <span>"overall"</span>: {
            <span>"overall_games_played"</span>: <span>82</span>,
            <span>"overall_team1_wins"</span>: <span>41</span>,
            <span>"overall_team2_wins"</span>: <span>24</span>,
            <span>"overall_draws"</span>: <span>17</span>,
            <span>"overall_team1_scored"</span>: <span>153</span>,
            <span>"overall_team2_scored"</span>: <span>99</span>
        },
        <span>"team1_at_home"</span>: {
            <span>"team1_games_played_at_home"</span>: <span>41</span>,
            <span>"team1_wins_at_home"</span>: <span>25</span>,
            <span>"team1_losses_at_home"</span>: <span>7</span>,
            <span>"team1_draws_at_home"</span>: <span>9</span>,
            <span>"team1_scored_at_home"</span>: <span>89</span>,
            <span>"team1_conceded_at_home"</span>: <span>42</span>
        },
        <span>"team2_at_home"</span>: {
            <span>"team2_games_played_at_home"</span>: <span>41</span>,
            <span>"team2_wins_at_home"</span>: <span>17</span>,
            <span>"team2_losses_at_home"</span>: <span>16</span>,
            <span>"team2_draws_at_home"</span>: <span>8</span>,
            <span>"team2_scored_at_home"</span>: <span>57</span>,
            <span>"team2_conceded_at_home"</span>: <span>64</span>
        }
    }
}

                </code>
            
```

#### QUERY PARAMETERS

| Field | Type | Description |
| --- | --- | --- |
| team\_1\_id | Integer | (required) First team by team\_id |
| team\_2\_id | Integer | (required) Second team by team\_id |

## Get Standing

```
                <code>
# Get Standing: Javascript Example Request

<span>async</span> <span><span>function</span> <span>getStanding</span>(<span></span>) </span>{

    <span>const</span> response = <span>await</span> fetch(<span>"https://api.soccerdataapi.com/standing/?league_id=228&amp;auth_token=320cae54d49a09f11c5cd23da43204a5543fb394"</span>, {
        <span>method</span>: <span>'GET'</span>,
        <span>headers</span>: {
            <span>"Content-Type"</span>: <span>"application/json"</span>,
            <span>"Accept-Encoding"</span>: <span>"gzip"</span>
        },
    })
    .then(<span><span>response</span> =&gt;</span> {
        <span>return</span> response;
    })
    .catch(<span><span>error</span> =&gt;</span> {
        <span>return</span> error;
    });

    <span>const</span> data = <span>await</span> response.json();
    <span>console</span>.log(data);

}
                </code>
            
```

```
                <code>
<span># Get Standing Curl Example Request</span>
curl --request GET \
  --compressed \
  --header <span>'Content-Type: application/json'</span>--url <span>'https://api.soccerdataapi.com/standing/?league_id=228&amp;auth_token=320cae54d49a09f11c5cd23da43204a5543fb394'</span> 
                </code>
            
```

```
                <code>
<span># Get Standing: Python Example Request</span>
<span>import</span> requests

url = <span>"https://api.soccerdataapi.com/standing/"</span>
querystring = {<span>'league_id'</span>: <span>228</span>, <span>'auth_token'</span>: <span>320</span>cae54d49a09f11c5cd23da43204a5543fb394}
headers = {
    <span>'Accept-Encoding'</span>: <span>'gzip'</span>,
    <span>'Content-Type'</span>: <span>'application/json'</span>
}
response = requests.get(url, headers=headers, params=querystring)
print(response.json())
                </code>
            
```

Retrieve standings by league id with a GET request to the endpoint:  
`https://api.soccerdataapi.com/standing/`

  

```
                <code>
Get Standing: Example JSON Response

{
    <span>"id"</span>: <span>228</span>,
    <span>"league"</span>: {
        <span>"id"</span>: <span>228</span>,
        <span>"name"</span>: <span>"Premier League"</span>
    },
    <span>"season"</span>: <span>"2023-2024"</span>,
    <span>"stage"</span>: [
        {
            <span>"stage_id"</span>: <span>6497</span>,
            <span>"stage_name"</span>: <span>"Regular Season"</span>,
            <span>"has_groups"</span>: <span>false</span>,
            <span>"is_active"</span>: <span>true</span>,
            <span>"standings"</span>: [
                {
                    <span>"position"</span>: <span>1</span>,
                    <span>"team_id"</span>: <span>3059</span>,
                    <span>"team_name"</span>: <span>"West Ham United"</span>,
                    <span>"position_attribute"</span>: <span>"Promotion - Champions League (Group Stage)"</span>,
                    <span>"games_played"</span>: <span>3</span>,
                    <span>"points"</span>: <span>7</span>,
                    <span>"wins"</span>: <span>2</span>,
                    <span>"draws"</span>: <span>1</span>,
                    <span>"losses"</span>: <span>0</span>,
                    <span>"goals_for"</span>: <span>7</span>,
                    <span>"goals_against"</span>: <span>3</span>
                },
                {
                    <span>"position"</span>: <span>2</span>,
                    <span>"team_id"</span>: <span>2909</span>,
                    <span>"team_name"</span>: <span>"Tottenham Hotspur"</span>,
                    <span>"position_attribute"</span>: <span>"Promotion - Champions League (Group Stage)"</span>,
                    <span>"games_played"</span>: <span>3</span>,
                    <span>"points"</span>: <span>7</span>,
                    <span>"wins"</span>: <span>2</span>,
                    <span>"draws"</span>: <span>1</span>,
                    <span>"losses"</span>: <span>0</span>,
                    <span>"goals_for"</span>: <span>6</span>,
                    <span>"goals_against"</span>: <span>2</span>
                },
                {
                    <span>"position"</span>: <span>3</span>,
                    <span>"team_id"</span>: <span>3068</span>,
                    <span>"team_name"</span>: <span>"Arsenal"</span>,
                    <span>"position_attribute"</span>: <span>"Promotion - Champions League (Group Stage)"</span>,
                    <span>"games_played"</span>: <span>3</span>,
                    <span>"points"</span>: <span>7</span>,
                    <span>"wins"</span>: <span>2</span>,
                    <span>"draws"</span>: <span>1</span>,
                    <span>"losses"</span>: <span>0</span>,
                    <span>"goals_for"</span>: <span>5</span>,
                    <span>"goals_against"</span>: <span>3</span>
                },

                ...

            ]
        },

        ...

    ]
}

                </code>
            
```

#### QUERY PARAMETERS

| Field | Type | Description |
| --- | --- | --- |
| league\_id | Integer | (required) Get standing by league\_id |
| season | String | (optional) Get standing by league season |

## Get Live Scores

```
                <code>
# Get Live Scores: Javascript Example Request

<span>async</span> <span><span>function</span> <span>getLivescores</span>(<span></span>) </span>{

    <span>const</span> response = <span>await</span> fetch(<span>"https://api.soccerdataapi.com/livescores/?auth_token=320cae54d49a09f11c5cd23da43204a5543fb394"</span>, {
        <span>method</span>: <span>'GET'</span>,
        <span>headers</span>: {
            <span>"Content-Type"</span>: <span>"application/json"</span>,
            <span>"Accept-Encoding"</span>: <span>"gzip"</span>
        },
    })
    .then(<span><span>response</span> =&gt;</span> {
        <span>return</span> response;
    })
    .catch(<span><span>error</span> =&gt;</span> {
        <span>return</span> error;
    });

    <span>const</span> data = <span>await</span> response.json();
    <span>console</span>.log(data);

}
                </code>
            
```

```
                <code>
<span># Get Live Scores Curl Example Request</span>
curl --request GET \
  --compressed \
  --header <span>'Content-Type: application/json'</span>--url <span>'https://api.soccerdataapi.com/livescores/?auth_token=320cae54d49a09f11c5cd23da43204a5543fb394'</span> 
                </code>
            
```

```
                <code>
<span># Get Live Scores: Python Example Request</span>
<span>import</span> requests

url = <span>"https://api.soccerdataapi.com/livescores/"</span>
querystring = {<span>'auth_token'</span>: <span>320</span>cae54d49a09f11c5cd23da43204a5543fb394}
headers = {
    <span>'Accept-Encoding'</span>: <span>'gzip'</span>,
    <span>'Content-Type'</span>: <span>'application/json'</span>
}
response = requests.get(url, headers=headers, params=querystring)
print(response.json())
                </code>
            
```

Retrieve live matches for current day (UTC) with a GET request to the endpoint:  
`https://api.soccerdataapi.com/livescores/`

  

```
                <code>
Get Live Scores: Example JSON Response

[
    {
        <span>"league_id"</span>: <span>206</span>,
        <span>"league_name"</span>: <span>"Liga Profesional"</span>,
        <span>"country"</span>: {
            <span>"id"</span>: <span>68</span>,
            <span>"name"</span>: <span>"argentina"</span>
        },
        <span>"is_cup"</span>: <span>false</span>,
        <span>"matches"</span>: [
            {
                <span>"id"</span>: <span>531585</span>,
                <span>"stage_id"</span>: <span>6347</span>,
                <span>"date"</span>: <span>"26/08/2023"</span>,
                <span>"time"</span>: <span>"00:30"</span>,
                <span>"teams"</span>: {
                    <span>"home"</span>: {
                        <span>"id"</span>: <span>3842</span>,
                        <span>"name"</span>: <span>"Colon"</span>
                    },
                    <span>"away"</span>: {
                        <span>"id"</span>: <span>3843</span>,
                        <span>"name"</span>: <span>"Gimnasia La Plata"</span>
                    }
                },
                <span>"stadium"</span>: {
                    <span>"id"</span>: <span>1891</span>,
                    <span>"name"</span>: <span>"Estadio Brigadier General Estanislao Lopez"</span>,
                    <span>"city"</span>: <span>"Ciudad de Santa Fe, Provincia de Santa Fe"</span>
                },
                <span>"status"</span>: <span>"finished"</span>,
                <span>"minute"</span>: <span>-1</span>,
                <span>"winner"</span>: <span>"home"</span>,
                <span>"has_extra_time"</span>: <span>false</span>,
                <span>"has_penalties"</span>: <span>false</span>,
                <span>"goals"</span>: {
                    <span>"home_ht_goals"</span>: <span>2</span>,
                    <span>"away_ht_goals"</span>: <span>0</span>,
                    <span>"home_ft_goals"</span>: <span>2</span>,
                    <span>"away_ft_goals"</span>: <span>0</span>,
                    <span>"home_et_goals"</span>: <span>-1</span>,
                    <span>"away_et_goals"</span>: <span>-1</span>,
                    <span>"home_pen_goals"</span>: <span>-1</span>,
                    <span>"away_pen_goals"</span>: <span>-1</span>
                },
                <span>"events"</span>: [
                    {
                        <span>"event_type"</span>: <span>"goal"</span>,
                        <span>"event_minute"</span>: <span>"14"</span>,
                        <span>"team"</span>: <span>"home"</span>,
                        <span>"player"</span>: {
                            <span>"id"</span>: <span>53675</span>,
                            <span>"name"</span>: <span>"J. Benítez"</span>
                        },
                        <span>"assist_player"</span>: <span>null</span>
                    },
                    {
                        <span>"event_type"</span>: <span>"goal"</span>,
                        <span>"event_minute"</span>: <span>"27"</span>,
                        <span>"team"</span>: <span>"home"</span>,
                        <span>"player"</span>: {
                            <span>"id"</span>: <span>53644</span>,
                            <span>"name"</span>: <span>"T. Galván"</span>
                        },
                        <span>"assist_player"</span>: <span>null</span>
                    },
                    {
                        <span>"event_type"</span>: <span>"yellow_card"</span>,
                        <span>"event_minute"</span>: <span>"30"</span>,
                        <span>"team"</span>: <span>"home"</span>,
                        <span>"player"</span>: {
                            <span>"id"</span>: <span>53590</span>,
                            <span>"name"</span>: <span>"F. Garcés"</span>
                        }
                    },

                    ...

                ],
                <span>"odds"</span>: {
                    <span>"match_winner"</span>: {
                        <span>"home"</span>: <span>1.84</span>,
                        <span>"draw"</span>: <span>3.5</span>,
                        <span>"away"</span>: <span>4.3</span>
                    },
                    <span>"over_under"</span>: {
                        <span>"total"</span>: <span>2.5</span>,
                        <span>"over"</span>: <span>2.1</span>,
                        <span>"under"</span>: <span>1.74</span>
                    },
                    <span>"handicap"</span>: {
                        <span>"market"</span>: <span>-0.5</span>,
                        <span>"home"</span>: <span>1.81</span>,
                        <span>"away"</span>: <span>1.96</span>
                    },
                    <span>"last_modified_timestamp"</span>: <span>1693017076</span>
                },
                <span>"lineups"</span>: {
                    <span>"lineup_type"</span>: <span>"live"</span>,
                    <span>"lineups"</span>: {
                        <span>"home"</span>: [
                            {
                                <span>"player"</span>: {
                                    <span>"id"</span>: <span>102150</span>,
                                    <span>"name"</span>: <span>"R. Botta"</span>
                                },
                                <span>"position"</span>: <span>"M"</span>
                            },
                            {
                                <span>"player"</span>: {
                                    <span>"id"</span>: <span>53656</span>,
                                    <span>"name"</span>: <span>"S. Moreyra"</span>
                                },
                                <span>"position"</span>: <span>"M"</span>
                            },

                            ...

                        ],
                        <span>"away"</span>: [
                            {
                                <span>"player"</span>: {
                                    <span>"id"</span>: <span>102150</span>,
                                    <span>"name"</span>: <span>"R. Botta"</span>
                                },
                                <span>"position"</span>: <span>"M"</span>
                            },
                            {
                                <span>"player"</span>: {
                                    <span>"id"</span>: <span>53656</span>,
                                    <span>"name"</span>: <span>"S. Moreyra"</span>
                                },
                                <span>"position"</span>: <span>"M"</span>
                            },

                            ...

                        ]
                    },
                    <span>"bench"</span>: {
                        <span>"home"</span>: [
                            {
                                <span>"player"</span>: {
                                    <span>"id"</span>: <span>53637</span>,
                                    <span>"name"</span>: <span>"B. Perlaza"</span>
                                },
                                <span>"position"</span>: <span>"M"</span>
                            },
                            {
                                <span>"player"</span>: {
                                    <span>"id"</span>: <span>53653</span>,
                                    <span>"name"</span>: <span>"L. Picco"</span>
                                },
                                <span>"position"</span>: <span>"M"</span>
                            },

                            ...

                        ],
                        <span>"away"</span>: [
                            {
                                <span>"player"</span>: {
                                    <span>"id"</span>: <span>53692</span>,
                                    <span>"name"</span>: <span>"Z. Zegarra"</span>
                                },
                                <span>"position"</span>: <span>"M"</span>
                            },
                            {
                                <span>"player"</span>: {
                                    <span>"id"</span>: <span>84921</span>,
                                    <span>"name"</span>: <span>"R. Saravia"</span>
                                },
                                <span>"position"</span>: <span>"M"</span>
                            },

                            ...

                        ]
                    },
                    <span>"sidelined"</span>: {
                        <span>"home"</span>: [
                            {
                                <span>"player"</span>: {
                                    <span>"id"</span>: <span>31889</span>,
                                    <span>"name"</span>: <span>"M. Novak"</span>
                                },
                                <span>"status"</span>: <span>"out"</span>,
                                <span>"desc"</span>: <span>"Injury"</span>
                            }
                        ],
                        <span>"away"</span>: [
                            {
                                <span>"player"</span>: {
                                    <span>"id"</span>: <span>31889</span>,
                                    <span>"name"</span>: <span>"M. Novak"</span>
                                },
                                <span>"status"</span>: <span>"out"</span>,
                                <span>"desc"</span>: <span>"Injury"</span>
                            }
                        ]
                    },
                    <span>"formation"</span>: {
                        <span>"home"</span>: <span>"4-3-3"</span>,
                        <span>"away"</span>: <span>"4-3-3"</span>
                    }
                },
                <span>"match_preview"</span>: {
                    <span>"has_preview"</span>: <span>true</span>,
                    <span>"word_count"</span>: <span>486</span>
                }
            },

            ...

        ]
    },

    ...

]

                </code>
            
```

## Get Matches

```
                <code>
# Get Matches: Javascript Example Request

<span>async</span> <span><span>function</span> <span>getMatches</span>(<span></span>) </span>{

    <span>const</span> response = <span>await</span> fetch(<span>"https://api.soccerdataapi.com/matches/?league_id=228&amp;auth_token=320cae54d49a09f11c5cd23da43204a5543fb394"</span>, {
        <span>method</span>: <span>'GET'</span>,
        <span>headers</span>: {
            <span>"Content-Type"</span>: <span>"application/json"</span>,
            <span>"Accept-Encoding"</span>: <span>"gzip"</span>
        },
    })
    .then(<span><span>response</span> =&gt;</span> {
        <span>return</span> response;
    })
    .catch(<span><span>error</span> =&gt;</span> {
        <span>return</span> error;
    });

    <span>const</span> data = <span>await</span> response.json();
    <span>console</span>.log(data);

}
                </code>
            
```

```
                <code>
<span># Get Matches Curl Example Request</span>
curl --request GET \
  --compressed \
  --header <span>'Content-Type: application/json'</span>--url <span>'https://api.soccerdataapi.com/matches/?league_id=228&amp;auth_token=320cae54d49a09f11c5cd23da43204a5543fb394'</span> 
                </code>
            
```

```
                <code>
<span># Get Matches: Python Example Request</span>
<span>import</span> requests

url = <span>"https://api.soccerdataapi.com/matches/"</span>
querystring = {<span>'league_id'</span>: <span>228</span>, <span>'auth_token'</span>: <span>320</span>cae54d49a09f11c5cd23da43204a5543fb394}
headers = {
    <span>'Accept-Encoding'</span>: <span>'gzip'</span>,
    <span>'Content-Type'</span>: <span>'application/json'</span>
}
response = requests.get(url, headers=headers, params=querystring)
print(response.json())
                </code>
            
```

Retrieve matches by date or league id (optionally with season paramater) with a GET request to the endpoint:  
`https://api.soccerdataapi.com/matches/`

  

```
                <code>
Get Matches: Example JSON Response

[
    {
        <span>"league_id"</span>: <span>228</span>,
        <span>"league_name"</span>: <span>"Premier League"</span>,
        <span>"country"</span>: {
            <span>"id"</span>: <span>8</span>,
            <span>"name"</span>: <span>"england"</span>
        },
        <span>"is_cup"</span>: <span>false</span>,
        <span>"matches"</span>: [
            {
                <span>"id"</span>: <span>567518</span>,
                <span>"stage"</span>: {
                    <span>"id"</span>: <span>6497</span>,
                    <span>"name"</span>: <span>"Premier League"</span>
                },
                <span>"date"</span>: <span>"11/08/2023"</span>,
                <span>"time"</span>: <span>"19:00"</span>,
                <span>"teams"</span>: {
                    <span>"home"</span>: {
                        <span>"id"</span>: <span>3104</span>,
                        <span>"name"</span>: <span>"Burnley"</span>
                    },
                    <span>"away"</span>: {
                        <span>"id"</span>: <span>4136</span>,
                        <span>"name"</span>: <span>"Manchester City"</span>
                    }
                },
                <span>"status"</span>: <span>"finished"</span>,
                <span>"minute"</span>: <span>-1</span>,
                <span>"winner"</span>: <span>"away"</span>,
                <span>"has_extra_time"</span>: <span>false</span>,
                <span>"has_penalties"</span>: <span>false</span>,
                <span>"goals"</span>: {
                    <span>"home_ht_goals"</span>: <span>0</span>,
                    <span>"away_ht_goals"</span>: <span>2</span>,
                    <span>"home_ft_goals"</span>: <span>0</span>,
                    <span>"away_ft_goals"</span>: <span>3</span>,
                    <span>"home_et_goals"</span>: <span>-1</span>,
                    <span>"away_et_goals"</span>: <span>-1</span>,
                    <span>"home_pen_goals"</span>: <span>-1</span>,
                    <span>"away_pen_goals"</span>: <span>-1</span>
                },
                <span>"odds"</span>: {
                    <span>"match_winner"</span>: {},
                    <span>"over_under"</span>: {},
                    <span>"handicap"</span>: {}
                },
                <span>"match_preview"</span>: {
                    <span>"has_previews"</span>: <span>false</span>,
                    <span>"word_count"</span>: <span>-1</span>
                }
            },

            ...

        ]
    }
}


                </code>
            
```

#### QUERY PARAMETERS

| Field | Type | Description |
| --- | --- | --- |
| date | String | (optionally required) Get matches by date |
| league\_id | Integer | (optionally required) Get matches by league\_id for current season |
| league\_id, season | String | (optional) Get matches by league and season |
| league\_id, date | String | (optional) Get matches by league and date |

## Get Match

```
                <code>
# Get Match: Javascript Example Request

<span>async</span> <span><span>function</span> <span>getMatch</span>(<span></span>) </span>{

    <span>const</span> response = <span>await</span> fetch(<span>"https://api.soccerdataapi.com/match/?match_id=531585&amp;auth_token=320cae54d49a09f11c5cd23da43204a5543fb394"</span>, {
        <span>method</span>: <span>'GET'</span>,
        <span>headers</span>: {
            <span>"Content-Type"</span>: <span>"application/json"</span>,
            <span>"Accept-Encoding"</span>: <span>"gzip"</span>
        },
    })
    .then(<span><span>response</span> =&gt;</span> {
        <span>return</span> response;
    })
    .catch(<span><span>error</span> =&gt;</span> {
        <span>return</span> error;
    });

    <span>const</span> data = <span>await</span> response.json();
    <span>console</span>.log(data);

}
                </code>
            
```

```
                <code>
<span># Get Match Curl Example Request</span>
curl --request GET \
  --compressed \
  --header <span>'Content-Type: application/json'</span>--url <span>'https://api.soccerdataapi.com/match/?match_id=531585&amp;auth_token=320cae54d49a09f11c5cd23da43204a5543fb394'</span> 
                </code>
            
```

```
                <code>
<span># Get Match: Python Example Request</span>
<span>import</span> requests

url = <span>"https://api.soccerdataapi.com/match/"</span>
querystring = {<span>'match_id'</span>: <span>531585</span>, <span>'auth_token'</span>: <span>320</span>cae54d49a09f11c5cd23da43204a5543fb394}
headers = {
    <span>'Accept-Encoding'</span>: <span>'gzip'</span>,
    <span>'Content-Type'</span>: <span>'application/json'</span>
}
response = requests.get(url, headers=headers, params=querystring)
print(response.json())
                </code>
            
```

Retrieve match by id with a GET request to the endpoint:  
`https://api.soccerdataapi.com/match/`

  

```
                <code>
Get Match: Example JSON Response

{
    <span>"id"</span>: <span>531585</span>,
    <span>"league"</span>: {
        <span>"id"</span>: <span>206</span>,
        <span>"name"</span>: <span>"Liga Profesional"</span>
    },
    <span>"stage"</span>: {
        <span>"id"</span>: <span>6347</span>,
        <span>"name"</span>: <span>"Liga Profesional Argentina: 2nd Phase"</span>
    },
    <span>"date"</span>: <span>"26/08/2023"</span>,
    <span>"time"</span>: <span>"00:30"</span>,
    <span>"teams"</span>: {
        <span>"home"</span>: {
            <span>"id"</span>: <span>3842</span>,
            <span>"name"</span>: <span>"Colon"</span>
        },
        <span>"away"</span>: {
            <span>"id"</span>: <span>3843</span>,
            <span>"name"</span>: <span>"Gimnasia La Plata"</span>
        }
    },
    <span>"stadium"</span>: {
        <span>"id"</span>: <span>1891</span>,
        <span>"name"</span>: <span>"Estadio Brigadier General Estanislao Lopez"</span>,
        <span>"city"</span>: <span>"Ciudad de Santa Fe, Provincia de Santa Fe"</span>
    },
    <span>"status"</span>: <span>"finished"</span>,
    <span>"minute"</span>: <span>-1</span>,
    <span>"winner"</span>: <span>"home"</span>,
    <span>"has_extra_time"</span>: <span>false</span>,
    <span>"has_penalties"</span>: <span>false</span>,
    <span>"goals"</span>: {
        <span>"home_ht_goals"</span>: <span>2</span>,
        <span>"away_ht_goals"</span>: <span>0</span>,
        <span>"home_ft_goals"</span>: <span>2</span>,
        <span>"away_ft_goals"</span>: <span>0</span>,
        <span>"home_et_goals"</span>: <span>-1</span>,
        <span>"away_et_goals"</span>: <span>-1</span>,
        <span>"home_pen_goals"</span>: <span>-1</span>,
        <span>"away_pen_goals"</span>: <span>-1</span>
    },
    <span>"events"</span>: [
        {
            <span>"event_type"</span>: <span>"goal"</span>,
            <span>"event_minute"</span>: <span>"14"</span>,
            <span>"team"</span>: <span>"home"</span>,
            <span>"player"</span>: {
                <span>"id"</span>: <span>53675</span>,
                <span>"name"</span>: <span>"J. Benítez"</span>
            },
            <span>"assist_player"</span>: <span>null</span>
        },
        {
            <span>"event_type"</span>: <span>"goal"</span>,
            <span>"event_minute"</span>: <span>"27"</span>,
            <span>"team"</span>: <span>"home"</span>,
            <span>"player"</span>: {
                <span>"id"</span>: <span>53644</span>,
                <span>"name"</span>: <span>"T. Galván"</span>
            },
            <span>"assist_player"</span>: <span>null</span>
        },
        
        ...

    ],
    <span>"odds"</span>: {
        <span>"match_winner"</span>: {
            <span>"home"</span>: <span>1.84</span>,
            <span>"draw"</span>: <span>3.5</span>,
            <span>"away"</span>: <span>4.3</span>
        },
        <span>"over_under"</span>: {
            <span>"total"</span>: <span>2.5</span>,
            <span>"over"</span>: <span>2.1</span>,
            <span>"under"</span>: <span>1.74</span>
        },
        <span>"handicap"</span>: {
            <span>"market"</span>: <span>-0.5</span>,
            <span>"home"</span>: <span>1.81</span>,
            <span>"away"</span>: <span>1.96</span>
        },
        <span>"last_modified_timestamp"</span>: <span>1693017076</span>
    },
    <span>"lineups"</span>: {
        <span>"lineup_type"</span>: <span>"live"</span>,
        <span>"lineups"</span>: {
            <span>"home"</span>: [
                {
                    <span>"player"</span>: {
                        <span>"id"</span>: <span>102150</span>,
                        <span>"name"</span>: <span>"R. Botta"</span>
                    },
                    <span>"position"</span>: <span>"M"</span>
                },
                {
                    <span>"player"</span>: {
                        <span>"id"</span>: <span>53656</span>,
                        <span>"name"</span>: <span>"S. Moreyra"</span>
                    },
                    <span>"position"</span>: <span>"M"</span>
                },
                
                ...

            ],
            <span>"away"</span>: [
                {
                    <span>"player"</span>: {
                        <span>"id"</span>: <span>102150</span>,
                        <span>"name"</span>: <span>"R. Botta"</span>
                    },
                    <span>"position"</span>: <span>"M"</span>
                },
                {
                    <span>"player"</span>: {
                        <span>"id"</span>: <span>53656</span>,
                        <span>"name"</span>: <span>"S. Moreyra"</span>
                    },
                    <span>"position"</span>: <span>"M"</span>
                },
                
                ...

            ]
        },
        <span>"bench"</span>: {
            <span>"home"</span>: [
                {
                    <span>"player"</span>: {
                        <span>"id"</span>: <span>53637</span>,
                        <span>"name"</span>: <span>"B. Perlaza"</span>
                    },
                    <span>"position"</span>: <span>"M"</span>
                },
                {
                    <span>"player"</span>: {
                        <span>"id"</span>: <span>53653</span>,
                        <span>"name"</span>: <span>"L. Picco"</span>
                    },
                    <span>"position"</span>: <span>"M"</span>
                },
                
                ...

            ],
            <span>"away"</span>: [
                {
                    <span>"player"</span>: {
                        <span>"id"</span>: <span>53692</span>,
                        <span>"name"</span>: <span>"Z. Zegarra"</span>
                    },
                    <span>"position"</span>: <span>"M"</span>
                },
                {
                    <span>"player"</span>: {
                        <span>"id"</span>: <span>84921</span>,
                        <span>"name"</span>: <span>"R. Saravia"</span>
                    },
                    <span>"position"</span>: <span>"M"</span>
                },
                
                ...

            ]
        },
        <span>"sidelined"</span>: {
            <span>"home"</span>: [
                {
                    <span>"player"</span>: {
                        <span>"id"</span>: <span>31889</span>,
                        <span>"name"</span>: <span>"M. Novak"</span>
                    },
                    <span>"status"</span>: <span>"out"</span>,
                    <span>"desc"</span>: <span>"Injury"</span>
                }
            ],
            <span>"away"</span>: [
                {
                    <span>"player"</span>: {
                        <span>"id"</span>: <span>31889</span>,
                        <span>"name"</span>: <span>"M. Novak"</span>
                    },
                    <span>"status"</span>: <span>"out"</span>,
                    <span>"desc"</span>: <span>"Injury"</span>
                }
            ]
        },
        <span>"formation"</span>: {
            <span>"home"</span>: <span>"4-3-3"</span>,
            <span>"away"</span>: <span>"4-3-3"</span>
        }
    },
    <span>"match_preview"</span>: {
        <span>"has_previews"</span>: <span>true</span>,
        <span>"word_count"</span>: <span>389</span>
    }
}

                </code>
            
```

#### QUERY PARAMETERS

| Field | Type | Description |
| --- | --- | --- |
| match\_id | Integer | (required) Get match by id |

## Get Match Preview

```
                <code>
# Get Match Preview: Javascript Example Request

<span>async</span> <span><span>function</span> <span>getMatchPreview</span>(<span></span>) </span>{

    <span>const</span> response = <span>await</span> fetch(<span>"https://api.soccerdataapi.com/match-preview/?match_id=544770&amp;auth_token=320cae54d49a09f11c5cd23da43204a5543fb394"</span>, {
        <span>method</span>: <span>'GET'</span>,
        <span>headers</span>: {
            <span>"Content-Type"</span>: <span>"application/json"</span>,
            <span>"Accept-Encoding"</span>: <span>"gzip"</span>
        },
    })
    .then(<span><span>response</span> =&gt;</span> {
        <span>return</span> response;
    })
    .catch(<span><span>error</span> =&gt;</span> {
        <span>return</span> error;
    });

    <span>const</span> data = <span>await</span> response.json();
    <span>console</span>.log(data);

}
                </code>
            
```

```
                <code>
<span># Get Match Preview Curl Example Request</span>
curl --request GET \
  --compressed \
  --header <span>'Content-Type: application/json'</span>--url <span>'https://api.soccerdataapi.com/match-preview/?match_id=544770&amp;auth_token=320cae54d49a09f11c5cd23da43204a5543fb394'</span>
                </code>
            
```

```
                <code>
<span># Get Match Preview: Python Example Request</span>
<span>import</span> requests

url = <span>"https://api.soccerdataapi.com/match-preview/"</span>
querystring = {<span>'match_id'</span>: <span>544770</span>, <span>'auth_token'</span>: <span>320</span>cae54d49a09f11c5cd23da43204a5543fb394}
headers = {
    <span>'Accept-Encoding'</span>: <span>'gzip'</span>,
    <span>'Content-Type'</span>: <span>'application/json'</span>
}
response = requests.get(url, headers=headers, params=querystring)
print(response.json())
                </code>
            
```

Retrieve match preview by match\_id with a GET request to the endpoint:  
`https://api.soccerdataapi.com/match-preview/`

  

```
                <code>
Get Match Preview: Example JSON Response

{
    <span>"match_id"</span>: <span>544770</span>,
    <span>"league"</span>: {
        <span>"id"</span>: <span>216</span>,
        <span>"name"</span>: <span>"Serie B"</span>,
        <span>"country"</span>: <span>"brazil"</span>
    },
    <span>"home"</span>: {
        <span>"id"</span>: <span>3958</span>,
        <span>"name"</span>: <span>"Chapecoense"</span>
    },
    <span>"away"</span>: {
        <span>"id"</span>: <span>3959</span>,
        <span>"name"</span>: <span>"Avai"</span>
    },
    <span>"word_count"</span>: <span>362</span>,
    <span>"date"</span>: <span>"27-08-2023"</span>,
    <span>"time"</span>: <span>"18:45"</span>,
    <span>"match_data"</span>: {
        <span>"weather"</span>: {
            <span>"temp_f"</span>: <span>62.1</span>,
            <span>"temp_c"</span>: <span>16.7</span>,
            <span>"description"</span>: <span>"sunny"</span>
        },
        <span>"excitement_rating"</span>: <span>5.53</span>,
        <span>"prediction"</span>: {
            <span>"type"</span>: <span>"match_winner"</span>,
            <span>"choice"</span>: <span>"Chapecoense Win"</span>
        }
    },
    <span>"content"</span>: [
        {
            <span>"name"</span>: <span>"p1"</span>,
            <span>"content"</span>: <span>"On Sunday, August 27, Chapecoense will face Avaí at Arena Condá Stadium in Chapecó, Santa Catarina, at 18:45 (UTC) in the Brazil Serie B league. This matchup marks a rematch of the teams' last game, a 1-4 win for Chapecoense in the Serie B back on May 13. Fans in attendance can expect sunny weather with a temperature of 62 degrees (F)."</span>
        },
        {
            <span>"name"</span>: <span>"h1"</span>,
            <span>"content"</span>: <span>"Kayke's Rewards Reaped After Two-Goal Match"</span>
        },
        {
            <span>"name"</span>: <span>"p2"</span>,
            <span>"content"</span>: <span>"Chapecoense have earned a total of 25 points in their last 24 matches, winning 6 and drawing 7 while losing 11. At home, they have won two and drawn one of their last five matches, but have lacked quality in attack, managing only 4 goals. Against similarly ranked opponents this year, they've had a difficult run, winning none, drawing one and losing three, scoring an average of 0.75 goals and conceding 1.75. In their last match, Kayke was instrumental in a 1-2 away win against Botafogo SP, where he scored both goals."</span>
        },
        {
            <span>"name"</span>: <span>"h2"</span>,
            <span>"content"</span>: <span>"Igor Bohn: The Reliable Road Goalkeeper with 10 Clean Sheets in 19 Games"</span>
        },
        {
            <span>"name"</span>: <span>"p3"</span>,
            <span>"content"</span>: <span>"Avaí come into this game in good form, having won two of their last five matches. In addition to their decent goal scoring record of 9 goals in their last five outings, they will be entertained with a full strength squad. Gabriel Poveda scored their lone goal when they drew 1-1 with CRB in the Serie B last time out. Their imperious form on the road has been led by their goalkeeper, Igor Bohn, who has kept ten clean sheets in 19 away matches this season."</span>
        },
        {
            <span>"name"</span>: <span>"p4"</span>,
            <span>"content"</span>: <span>"In the past 10 matches between Chapecoense and Avaí, an average of 2.3 goals have been scored. Out of 59 head-to-head meetings, Chapecoense has won 29, drawn 11 and Avaí has won 19 times."</span>
        },
        {
            <span>"name"</span>: <span>"h3"</span>,
            <span>"content"</span>: <span>"Chapecoense Seeks to Leapfrog Avaí on League Table"</span>
        },
        {
            <span>"name"</span>: <span>"p5"</span>,
            <span>"content"</span>: <span>"Chapecoense sit 1 point behind Avaí in the league table and have a chance to jump them with their next game. This gives the home team plenty of incentive to give it their all and strive to get the result they need. Going into the match, Chapecoense will be highly motivated to score goals and solidify their place at the top."</span>
        }
    ]
}

                </code>
            
```

#### QUERY PARAMETERS

| Field | Type | Description |
| --- | --- | --- |
| match\_id | Integer | (required) Get preview by match\_id |

## Get Upcoming Match Previews

```
                <code>
# Get Upcoming Match Previews: Javascript Example Request

<span>async</span> <span><span>function</span> <span>getUpcomingMatchPreviews</span>(<span></span>) </span>{

    <span>const</span> response = <span>await</span> fetch(<span>"https://api.soccerdataapi.com/match-previews-upcoming/?auth_token=320cae54d49a09f11c5cd23da43204a5543fb394"</span>, {
        <span>method</span>: <span>'GET'</span>,
        <span>headers</span>: {
            <span>"Content-Type"</span>: <span>"application/json"</span>,
            <span>"Accept-Encoding"</span>: <span>"gzip"</span>
        },
    })
    .then(<span><span>response</span> =&gt;</span> {
        <span>return</span> response;
    })
    .catch(<span><span>error</span> =&gt;</span> {
        <span>return</span> error;
    });

    <span>const</span> data = <span>await</span> response.json();
    <span>console</span>.log(data);

}
                </code>
            
```

```
                <code>
<span># Get Upcoming Match Previews Curl Example Request</span>
curl --request GET \
  --compressed \
  --header <span>'Content-Type: application/json'</span>--url <span>'https://api.soccerdataapi.com/match-previews-upcoming/?auth_token=320cae54d49a09f11c5cd23da43204a5543fb394'</span> 
                </code>
            
```

```
                <code>
<span># Get Upcoming Match Previews: Python Example Request</span>
<span>import</span> requests

url = <span>"https://api.soccerdataapi.com/match-previews-upcoming/"</span>
querystring = {<span>'auth_token'</span>: <span>320</span>cae54d49a09f11c5cd23da43204a5543fb394}
headers = {
    <span>'Accept-Encoding'</span>: <span>'gzip'</span>,
    <span>'Content-Type'</span>: <span>'application/json'</span>
}
response = requests.get(url, headers=headers, params=querystring)
print(response.json())
                </code>
            
```

Retrieve upcoming match previews with a GET request to the endpoint:  
`https://api.soccerdataapi.com/match-previews-upcoming/`

  

```
                <code>
Get Upcoming Match Previews: Example JSON Response

[
    {
        <span>"league_id"</span>: <span>216</span>,
        <span>"league_name"</span>: <span>"Serie B"</span>,
        <span>"country"</span>: {
            <span>"id"</span>: <span>67</span>,
            <span>"name"</span>: <span>"brazil"</span>
        },
        <span>"is_cup"</span>: <span>false</span>,
        <span>"match_previews"</span>: [
            {
                <span>"id"</span>: <span>544770</span>,
                <span>"date"</span>: <span>"27/08/2023"</span>,
                <span>"time"</span>: <span>"18:45"</span>,
                <span>"teams"</span>: {
                    <span>"home"</span>: {
                        <span>"id"</span>: <span>3958</span>,
                        <span>"name"</span>: <span>"Chapecoense"</span>
                    },
                    <span>"away"</span>: {
                        <span>"id"</span>: <span>3959</span>,
                        <span>"name"</span>: <span>"Avai"</span>
                    }
                },
                <span>"word_count"</span>: <span>362</span>,  
            },

            ...

        ]
    },

    ...

]


                </code>
            
```

## Errors

```
                <code>
<span># Invalid Request</span>
                </code>
            
```

Invalid requests respond with a 200 status code, and an error message found in the 'detail' attribute:

`{"detail": "Invalid token."}`

`{"detail": "Request was throttled. Expected available in 60 seconds."}`

`{"detail": "Error fetching match.""}`

  

```
                <code>
Error: Example JSON Response

{
    'detail': 'Error fetching match.'
}


                </code>
            
```