/**
 * Copyright (c) 2016 Weitian LI <liweitianux@live.com>
 * MIT license
 *
 * JavaScript codes for the Web UI of "fg21sim"
 */


$(document).ready(function () {
  /**
   * Offset the page to adjust for the fixed navigation banner
   */
  var nav_height = $("nav.navigation").outerHeight();

  var scroll_target = function () {
    if ($(":target").length) {
      var offset = $(":target").offset();
      var scroll_to = offset.top - nav_height * 1.2;
      $("html, body").animate({scrollTop: scroll_to}, 0);
    }
  };

  $(window).on("hashchange", scroll_target);
  /* FIXME: This seems not work ... */
  if (window.location.hash) {
    scroll_target();
  }
});
