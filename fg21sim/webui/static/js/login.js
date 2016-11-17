/**
 * Copyright (c) 2016 Weitian LI <liweitianux@live.com>
 * MIT license
 *
 * JavaScript codes for the Web UI of "fg21sim"
 */

"use strict";


$(document).ready(function () {
  /**
   * Submit the login form on "Enter" key
   * Credit: https://stackoverflow.com/a/12518467/4856091
   */
  $("form#login input").keypress(function (e) {
    if (e.which === 13) {
      // "Enter" key pressed
      $(e.target).closest("form").submit();
      return false;
    }
  });
});
