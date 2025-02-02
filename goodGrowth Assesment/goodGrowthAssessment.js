(function () {
  // get the __NEXT_DATA__ element from the page
  var nextDataElem = document.getElementById("__NEXT_DATA__");
  if (!nextDataElem) {
    console.error("No __NEXT_DATA__ element found.");
    return;
  }

  // Try to parse the JSON data from the element
  var coordinates = null;
  try {
    var jsonData = JSON.parse(nextDataElem.textContent);

    // A simple recursive function to look for latitude and longitude inside the JSON
    function findCoords(obj) {
      // Check if obj exists and is an object
      if (obj && typeof obj === "object") {
        // If both latitude and longitude are properties, return them
        if (obj.hasOwnProperty("latitude") && obj.hasOwnProperty("longitude")) {
          return { latitude: obj.latitude, longitude: obj.longitude };
        }
        // loop through every property
        for (var prop in obj) {
          if (obj.hasOwnProperty(prop)) {
            var result = findCoords(obj[prop]);
            if (result) {
              return result;
            }
          }
        }
      }
      return null;
    }

    coordinates = findCoords(jsonData);
  } catch (err) {
    console.error("error parsing data", err);
    return;
  }

  // If coordinates are not found, return error
  if (
    !coordinates ||
    !coordinates.latitude ||
    !coordinates.longitude
  ) {
    console.error("could not extract coordinates");
    return;
  }

  // Convert the coordinates to floats
  var lat = parseFloat(coordinates.latitude);
  var lon = parseFloat(coordinates.longitude);
  console.log("Coordinates from __NEXT_DATA__:", lat, lon);

  // Build the weather API URL using the coordinates
  var apiKey = "a2ef86c41a"; 
  var weatherApiUrl =
    "https://europe-west1-amigo-actions.cloudfunctions.net/recruitment-mock-weather-endpoint/forecast?appid=" +
    apiKey +
    "&lat=" +
    lat +
    "&lon=" +
    lon;

  // Fetch the weather data from the API
  fetch(weatherApiUrl)
    .then(function (response) {
      if (!response.ok) {
        throw new Error("HTTP error! Status: " + response.status);
      }
      return response.json();
    })
    .then(function (data) {
      console.log("connect successfully to api:", data);
      if (!data || !data.list || data.list.length === 0) {
        console.error("failure to connect to api:", data);
        return;
      }

      // Create a container for the weather widget
      var forecasts = data.list;
      var widget = document.createElement("section");
      widget.className = "weather-widget";

      // Create a row to hold the forecast columns
      var row = document.createElement("div");
      row.className = "weather-row";

      // Loop through each forecast and collect data
      for (var i = 0; i < forecasts.length; i++) {
        var forecast = forecasts[i];

        // Get date and time
        var dtTxt = forecast.dt_txt;
        var dtParts = dtTxt.split(" ");
        var datePart = dtParts[0];
        var timePart = dtParts[1];

        // Get all the main data
        var temp = forecast.main.temp;
        var feelsLike = forecast.main.feels_like;
        var tempMin = forecast.main.temp_min;
        var tempMax = forecast.main.temp_max;
        var pressure = forecast.main.pressure;
        var seaLevel = forecast.main.sea_level;
        var grndLevel = forecast.main.grnd_level;
        var humidity = forecast.main.humidity;
        var tempKf = forecast.main.temp_kf;

        // Get weather details
        var weather = (forecast.weather && forecast.weather.length > 0)
          ? forecast.weather[0]
          : null;
        var weatherMain = weather ? weather.main : "N/A";
        var weatherDesc = weather ? weather.description : "N/A";
        var weatherIcon = weather ? weather.icon : "";
        var weatherIconUrl = weatherIcon
          ? "https://openweathermap.org/img/wn/" + weatherIcon + "@2x.png"
          : "";
        var weatherId = weather ? weather.id : "N/A";

        // Get additional data
        var cloudsAll = forecast.clouds ? forecast.clouds.all : "N/A";
        var windSpeed = forecast.wind.speed;
        var windDeg = forecast.wind.deg;
        var windGust = forecast.wind.gust;
        var visibility = forecast.visibility;
        var pop = forecast.pop;
        var rain3h =
          forecast.rain && forecast.rain["3h"]
            ? forecast.rain["3h"]
            : 0;
        var pod =
          forecast.sys && forecast.sys.pod
            ? forecast.sys.pod
            : "N/A";

        //column element for a forecast
        var column = document.createElement("div");
        column.className = "weather-column";

        // Combining all the data
        column.innerHTML =
          "<div class='weather-datetime'>" +
            "<div class='weather-date'><strong>Date:</strong> " +
              datePart +
            "</div>" +
            "<div class='weather-time'><strong>Time:</strong> " +
              timePart +
            "</div>" +
          "</div>" +
          "<div class='weather-icon-block'>" +
            "<img src='" + weatherIconUrl + "' alt='" + weatherDesc + "' class='weather-icon'>" +
          "</div>" +
          "<div class='weather-info'><strong>Weather ID:</strong> " +
            weatherId +
          "</div>" +
          "<div class='weather-info'><strong>Main:</strong> " +
            weatherMain +
          "</div>" +
          "<div class='weather-info'><strong>Description:</strong> " +
            weatherDesc +
          "</div>" +
          "<div class='weather-info'><strong>Temp:</strong> " +
            temp +
          "</div>" +
          "<div class='weather-info'><strong>Feels Like:</strong> " +
            feelsLike +
          "</div>" +
          "<div class='weather-info'><strong>Temp Min:</strong> " +
            tempMin +
          "</div>" +
          "<div class='weather-info'><strong>Temp Max:</strong> " +
            tempMax +
          "</div>" +
          "<div class='weather-info'><strong>Pressure:</strong> " +
            pressure +
          "</div>" +
          "<div class='weather-info'><strong>Sea Level:</strong> " +
            seaLevel +
          "</div>" +
          "<div class='weather-info'><strong>Ground Level:</strong> " +
            grndLevel +
          "</div>" +
          "<div class='weather-info'><strong>Humidity:</strong> " +
            humidity +
          "</div>" +
          "<div class='weather-info'><strong>Temp KF:</strong> " +
            tempKf +
          "</div>" +
          "<div class='weather-info'><strong>Clouds:</strong> " +
            cloudsAll +
          "</div>" +
          "<div class='weather-info'><strong>Wind Speed:</strong> " +
            windSpeed +
          "</div>" +
          "<div class='weather-info'><strong>Wind Deg:</strong> " +
            windDeg +
          "</div>" +
          "<div class='weather-info'><strong>Wind Gust:</strong> " +
            windGust +
          "</div>" +
          "<div class='weather-info'><strong>Visibility:</strong> " +
            visibility +
          "</div>" +
          "<div class='weather-info'><strong>POP:</strong> " +
            pop +
          "</div>" +
          "<div class='weather-info'><strong>Rain (3h):</strong> " +
            rain3h +
          "</div>" +
          "<div class='weather-info'><strong>Pod:</strong> " +
            pod +
          "</div>";

        // columns of data for rows
        row.appendChild(column);
      }

      // row foir weather forecast instide widgets
      widget.appendChild(row);

      // CSS styling
      var styleElem = document.createElement("style");
      styleElem.innerHTML =
        `
        /* Hand-tuned styles for the weather widget */
        .weather-widget {
          font-family: 'Helvetica Neue', Arial, sans-serif;
          max-width: 90%;
          margin: 20px auto;
          padding: 20px;
          background-color: #fff;
          border: 1px solid #ddd;
          border-radius: 8px;
          box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        .weather-row {
          display: flex;
          overflow-x: auto;
          gap: 15px;
          padding-bottom: 10px;
        }
        .weather-column {
          flex: 0 0 220px;
          background: #f9f9f9;
          padding: 10px;
          border: 1px solid #eee;
          border-radius: 6px;
          box-shadow: 1px 1px 4px rgba(0,0,0,0.1);
        }
        .weather-datetime {
          margin-bottom: 8px;
        }
        .weather-date, .weather-time {
          font-size: 14px;
          color: #333;
          margin: 4px 0;
        }
        .weather-icon-block {
          text-align: center;
          margin-bottom: 8px;
        }
        .weather-icon {
          width: 50px;
          height: 50px;
        }
        .weather-info {
          font-size: 12px;
          color: #555;
          margin: 2px 0;
        }
        /* Responsive tweaks for smaller screens */
        @media (max-width: 600px) {
          .weather-column {
            flex: 0 0 160px;
          }
        }
      `;
      document.head.appendChild(styleElem);

      // Try to insert the widget above the check availability title
      var checkAvailabilityElem = document.querySelector(
        "h2.Typographystyle__HeadingLevel2-sc-86wkop-1.ezdckS"
      );
      if (checkAvailabilityElem && checkAvailabilityElem.parentNode) {
        checkAvailabilityElem.parentNode.insertBefore(widget, checkAvailabilityElem);
      } else {
        // otherwise add it to the top of the body
        document.body.insertBefore(widget, document.body.firstChild);
      }

      console.log("weather widget added successfully");
    })
    .catch(function (error) {
      console.error("error fetching weather data:", error);
    });
})();
