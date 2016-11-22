/**
 * Copyright (c) 2016 Weitian LI <liweitianux@live.com>
 * MIT license
 *
 * Web UI of "fg21sim"
 * Manage and manipulate the simulation products
 */

"use strict";


/**
 * Show notification contents in the "#modal-products" modal box.
 */
var showModalProducts = function (data) {
  var modalBox = $("#modal-products");
  showModal(modalBox, data);
};


/**
 * Reset the manifest table
 */
var resetManifestTable = function () {
  var table = $("table#products-manifest");
  // Empty table caption, head, and body
  table.children().empty();
};


/**
 * Compose the manifest table cell element for each product
 *
 * @param {Object} data - The metadata of the product
 * @param {Boolean} localhost - Whether the connection from localhost
 */
var makeManifestTableCell = function (data, localhost) {
  var cell = $("<td>").addClass("product");
  cell.data("localhost", localhost)
    .data("frequency", data.frequency)
    .data("path", data.healpix.path)
    .data("size", data.healpix.size)
    .data("md5", data.healpix.md5);
  cell.append($("<span>").attr("title", "Show product information")
              .addClass("info btn btn-small fa fa-info-circle"));
  cell.append($("<span>").attr("title", "Download HEALPix map")
              .addClass("healpix btn btn-small fa fa-file"));
  if (data.hpx) {
    cell.data("hpx-image", true)
      .data("hpx-path", data.hpx.path)
      .data("hpx-size", data.hpx.size)
      .data("hpx-md5", data.hpx.md5);
    cell.append($("<span>").attr("title", localhost
                                 ? "Open HPX FITS image"
                                 : "Download HPX FITS image")
                .addClass("hpx btn btn-small fa fa-image"));
  } else {
    cell.data("hpx-image", false);
    cell.append($("<span>").attr("title", "Generate HPX image")
                .addClass("hpx hpx-convert btn btn-small fa fa-image"));
  }
  return cell;
};


/**
 * Load the products manifest into table
 *
 * @param {Object} manifest - The products manifest data
 * @param {Boolean} [localhost=false] - Whether the connection from localhost
 */
var loadManifestToTable = function (manifest, localhost) {
  localhost = typeof localhost !== "undefined" ? localhost : false;

  var frequency = manifest.frequency;
  var components = Object.keys(manifest);
  // Remove the `frequency` element
  components.splice(components.indexOf("frequency"), 1);

  // Reset the table first
  resetManifestTable();

  var table = $("table#products-manifest");
  var row;
  var cell;
  // Table head
  row = $("<tr>");
  row.append($("<th scope='col'>").text("Frequency (" + frequency.unit + ")"));
  components.forEach(function (compID) {
    row.append($("<th scope='col'>").text(compID).data("compID", compID));
  });
  table.children("thead").append(row);

  // Table body
  var tbody = table.children("tbody");
  frequency.id.forEach(function (freqID) {
    // One table row for each frequency
    row = $("<tr>");
    row.append($("<th scope='row'>").addClass("frequency freqid-" + freqID)
              .text(frequency.frequencies[freqID])
              .data("freqID", freqID));
    components.forEach(function (compID) {
      cell = makeManifestTableCell(manifest[compID][freqID], localhost);
      cell.data("compID", compID).data("freqID", freqID);
      row.append(cell);
    });
    tbody.append(row);
  });
};


/**
 * Update the content of one single cell in the manifest table
 */
var updateManifestTableCell = function (compID, freqID, data) {
  var tr = $("table#products-manifest").find("tr.frequency.freqid-" + freqID);
  tr.find("td").each(function () {
    if ($(this).data("compID") === compID) {
      // Found the target table cell
      $(this).html(makeManifestTableCell(data));
    }
  });
};


/**
 * Request the server to load the specified products manifest
 *
 * @param {String} url - The URL that handles the "products" AJAX requests.
 * @param {String} manifestfile - The (absolute) path to the manifest file.
 */
var loadServerManifest = function (url, manifestfile) {
  if (typeof manifestfile === "undefined") {
    manifestfile = $("input#products-manifest").val();
  };
  var data = {action: "load", manifestfile: manifestfile};
  return $.postJSON(url, data)
    .fail(function (jqxhr) {
      var modalData = {};
      modalData.icon = "times-circle";
      modalData.contents = "Failed to load the products manifest!";
      modalData.code = jqxhr.status;
      modalData.reason = jqxhr.statusText;
      showModalProducts(modalData);
    });
};


/**
 * Save the current products manifest to file
 *
 * @param {Boolean} [clobber=false] - Whether overwrite the existing file.
 */
var saveServerManifest = function (url, clobber) {
  clobber = typeof clobber !== "undefined" ? clobber : false;
  var outfile = $("input#products-manifest").val();
  var data = {action: "save", outfile: outfile, clobber: clobber};
  return $.postJSON(url, data)
    .done(function () {
      var modalData = {};
      modalData.icon = "check-circle";
      modalData.contents = "Current products manifest saved.";
      showModalProducts(modalData);
    })
    .fail(function (jqxhr) {
      var modalData = {};
      modalData.icon = "times-circle";
      modalData.contents = "Failed to save current products manifest!";
      modalData.code = jqxhr.status;
      modalData.reason = jqxhr.statusText;
      showModalProducts(modalData);
    });
};


/**
 * Get the products manifest from server
 */
var getServerManifest = function (url) {
  return $.getJSON(url)
    .fail(function (jqxhr) {
      var modalData = {};
      modalData.icon = "times-circle";
      modalData.contents = "Failed to load the products manifest!";
      modalData.code = jqxhr.status;
      modalData.reason = jqxhr.statusText;
      showModalProducts(modalData);
    });
};


/**
 * Reset the products manifest on both the server side and client side
 */
var resetManifest = function (url) {
  $.postJSON(url, {action: "reset"})
    .done(function () {
      resetManifestTable();
      // Popup a modal notification
      var modalData = {};
      modalData.icon = "check-circle";
      modalData.contents = "Reset the products manifest.";
      showModalProducts(modalData);
    })
    .fail(function (jqxhr) {
      var modalData = {};
      modalData.icon = "times-circle";
      modalData.contents = "Failed to reset the products manifest on server!";
      modalData.code = jqxhr.status;
      modalData.reason = jqxhr.statusText;
      showModalProducts(modalData);
    });
};


/**
 * Convert a product from HEALPix map to HPX projected FITS image
 *
 * @param {String} compID - ID of the foreground component
 * @param {Integer} freqID - Frequency ID
 */
var convertProductHPX = function (url, compID, freqID) {
  ;
};


$(document).ready(function () {
  // URL to handle the "products" AJAX requests
  var ajax_url = "/ajax/products";

  // Update the products manifest file path
  $("#conf-form input[name='workdir'], " +
    "#conf-form input[name='common/manifest']").on("change", function (e) {
      var workdir = $("#conf-form input[name='workdir']").val();
      var manifest = $("#conf-form input[name='output/manifest']").val();
      $("input#products-manifest").val(joinPath(workdir, manifest));
  });

  // Get and load products manifest into table
  $("#load-products").on("click", function () {
    loadServerManifest(ajax_url)
      .then(function () {
        return getServerManifest(ajax_url);
      })
      .done(function (response) {
        console.log("GET products response:", response);
        var modalData = {};
        if ($.isEmptyObject(response.manifest)) {
          modalData.icon = "warning";
          modalData.contents = "Products manifest not loaded on the server.";
          showModalProducts(modalData);
        } else {
          loadManifestToTable(response.manifest, response.localhost);
          modalData.icon = "check-circle";
          modalData.contents = "Loaded products manifest to table.";
          showModalProducts(modalData);
        }
      });
  });

  // Save current products manifest
  $("#save-products").on("click", function () {
    saveServerManifest(ajax_url);
  });

  // Show product information (metadata)
  // NOTE:
  // Since the table cell are dynamically built, so "click" event handler
  // should be bound to the "document" with the proper selector.
  $(document).on("click", "td.product > .info", function () {
    var product = $(this).closest("td");
    var modalData = {
      icon: "info-circle",
      title: ("Product: " + product.data("compID") + " @ " +
              product.data("frequency") + " (#" + product.data("freqID") + ")"),
      contents: [
        ("<strong>HEALPix map:</strong> " + product.data("path") + ", size: " +
         product.data("size") + " bytes, MD5: " + product.data("md5"))
      ]
    };
    if (product.data("hpx-image")) {
      var p = ("HPX image: " + product.data("hpx-path") + ", size: " +
               product.data("hpx-size") + "bytes, MD5: " +
               product.data("hpx-md5"));
      modalData.contents.push(p);
    }
    showModalProducts(modalData);
  });
});
