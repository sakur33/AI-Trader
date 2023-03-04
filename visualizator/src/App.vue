<template>
  <section>
    <div class="wrapper" :class="showSide ? 'no-overflow' : ''" id="wrapper">
      <sc-header
        @click-burger-desktop="clickBurgerDesktop"
        @click-burger-mobile="clickBurgerMobile"
      />
      <router-view
        class="app__content"
        :class="!expandSide ? 'content-expanded' : 'content-reduced'"
      />
      <div
        v-if="showSide"
        class="hidden-layer"
        id="hidden-layer"
        @click="hideMenu"
      />
      <sc-sidebar :showSB="showSide" :expandSB="expandSide" />
    </div>
  </section>
</template>

<script lang="ts">
import { Component, Vue } from "vue-property-decorator";
import Sidebar from "./components/SidebarComponent.vue";
import Header from "./components/HeaderComponent.vue";

@Component({
  components: {
    "sc-header": Header,
    "sc-sidebar": Sidebar,
  },
})
export default class App extends Vue {
  private expandSide = !(window.innerWidth < 1367);

  private showSide = false;

  public hideMenu() {
    this.showSide = false;
  }

  public clickBurgerDesktop() {
    this.expandSide = !this.expandSide;
  }

  public clickBurgerMobile() {
    if (window.innerWidth < 1024) {
      this.expandSide = true;
    }
    this.showSide = true;
  }
}
</script>

<style lang="scss">
@import "~bulma/sass/utilities/_all";

$family-primary: Inter, BlinkMacSystemFont, "Segoe UI", Roboto, Oxygen, Ubuntu,
  Cantarell, "Open Sans", "Helvetica Neue", sans-serif;
$body-font-size: 14px;

@import "~bulma";
@import "~buefy/src/scss/buefy";

.carousel {
  .carousel-items {
    overflow: visible;
  }
}

* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
  scrollbar-width: thin;
  scrollbar-color: #c3c3c3 #f4f4f4;
}
html {
  background: #f5f4f6 !important;
}
*::-webkit-scrollbar {
  width: 6px;
}

*::-webkit-scrollbar-track {
  background: #f4f4f4;
  border-radius: 3px;
}

*::-webkit-scrollbar-thumb {
  background: #c3c3c3;
  border-radius: 3px;
}

*::-webkit-scrollbar-thumb:hover {
  background: rgb(100, 100, 100);
  border-radius: 3px;
}

*::-webkit-scrollbar-thumb:active {
  background: rgb(68, 68, 68);
  border-radius: 3px;
}
body {
  color: rgb(87, 87, 87);
}

.app__content {
  width: 100%;
  margin: 0;
  padding: 5px;
  // justify-content: space-between;
}

.wrapper {
  width: 100%;
  height: 100vh;
  display: grid;
  grid-template-areas:
    "sidebar header"
    "sidebar app__content";
  grid-template-columns: min-content 1fr;
  grid-template-rows: 5rem 1fr;
}

.margin-5 {
  margin-left: 20px;
  color: rgb(63, 63, 63);
}

.carousel-arrow .icon.has-icons-left,
.carousel-arrow .icon.has-icons-right {
  position: fixed !important;
}

.content-reduced {
  .carousel-arrow .icon.has-icons-left {
    left: calc(233px + 25px) !important;
  }
}

.content-expanded {
  .carousel-arrow .icon.has-icons-left {
    left: calc(80px + 25px) !important;
  }
}

@media (max-width: 1366px) {
  .tags > .tag {
    font-size: 0.55rem;
  }
}

@media (max-width: 1023px) {
  .wrapper {
    grid-template-areas:
      "header"
      "app__content";
    grid-template-columns: minmax(0, 1fr) !important;
  }

  .tags > .tag {
    font-size: 0.6rem;
  }

  .modal-card {
    width: 90vw;
    margin: auto !important;
  }

  .buttons:last-child {
    margin-bottom: auto !important;
  }

  .no-overflow {
    overflow: hidden;
  }

  .hidden-layer {
    background: rgb(236, 236, 236, 0.5);
    width: 100%;
    height: 100vh;
    position: absolute;
    z-index: 2;
    overflow: hidden;
  }

  .sidebar {
    position: absolute;
    z-index: 3;
    transition: width 0.5s;
  }

  .carousel-items {
    overflow: scroll !important;
  }

  .content-reduced {
    .carousel-arrow .icon.has-icons-left {
      left: 0.7rem !important;
    }
    .carousel-arrow .icon.has-icons-right {
      right: 0.7rem !important;
    }
  }
}

@media (min-width: 1024px) {
  .menu-desktop {
    display: block;
  }
  .menu-mobile {
    display: none;
  }
  .hidden-layer {
    display: none !important;
  }

  .sidebar {
    display: flex !important;
    height: 100vh;
  }

  .app__content {
    overflow: scroll;
  }
}

@media (min-width: 1366px) {
  .app__content {
    padding-left: 3vw;
    padding-right: 1vw;
  }
}

@media only screen and (min-resolution: 120dpi) and (orientation: landscape) {
  html {
    font-size: 80%;
  }
  .top .chart-title {
    font-size: 1rem !important;
  }
}

@media screen and (max-width: 1023px) {
  .dropdown.is-mobile-modal > .dropdown-menu {
    overflow-y: unset !important;
  }
  .text-holder {
    max-height: 80vh;
    overflow-y: scroll;
  }
}
</style>
