<template>
  <div class="header">
    <div class="left-header">
      <b-icon
        icon="ellipsis-v"
        pack="fa"
        size="is-small"
        class="dots menu-mobile"
        @click.native="clickBM"
      ></b-icon>
      <b-icon
        icon="ellipsis-v"
        pack="fa"
        size="is-small"
        class="dots menu-desktop"
        @click.native="clickBD"
      ></b-icon>
      <div class="breadcrumbs" id="breadcrumbs">
        {{ breadcrumb }}
      </div>
    </div>
  </div>
</template>

<script lang="ts">
import { Component, Vue } from "vue-property-decorator";

@Component({})
export default class Header extends Vue {
  private clickBD() {
    this.$emit("click-burger-desktop");
  }

  private clickBM() {
    this.$emit("click-burger-mobile");
  }

  public capitalizeFirstLetter = (str: string) =>
    str.length > 0 ? str.charAt(0).toUpperCase() + str.slice(1) : "";

  private get breadcrumb() {
    return this.capitalizeFirstLetter(`${this.$route.name}`.toString());
  }
}
</script>

<style lang="scss" scoped>
.header {
  grid-area: header;
  background: #f5f4f6;
  border-bottom: 1px solid rgb(230, 224, 224, 0.35);
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin: 10px 30px 10px 10px;
}

.left-header {
  display: flex;
  flex-direction: row;
  align-items: center;
  width: 50%;
  margin-left: 10px;
}

.right-header {
  display: flex;
  flex-direction: row;
  align-items: center;
  justify-content: flex-end;
  width: 50%;
  margin-right: 10px;
}

.dropdown {
  margin-left: 15px !important;
}

.dots,
.breadcrumbs {
  background-color: #f9f9f9;
  color: rgb(63, 63, 63);
  border-radius: 7px;
  padding: 0px 10px;
  width: auto;
  font-size: 0.8em;
  cursor: pointer;
}

.breadcrumbs {
  font-size: 14px;
  margin-left: 10px;
}

.mdi {
  font-size: 1.12rem;
  color: rgba(87, 87, 87, 0.85);
  cursor: pointer;
}

.media {
  align-items: center;
  height: 6em;
}

.logout,
.menu-item-with-ico {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 3rem;
}

.logout,
a.dropdown-item.logout:hover {
  color: #e63148;
}

.ellipsis {
  text-overflow: ellipsis;
  /* Required for text-overflow to do anything */
  white-space: nowrap;
  overflow: hidden;
}

strong {
  font-size: 1.1rem;
}

.media-content {
  min-width: 0;
}

.text-holder {
  overflow-y: auto;
  max-height: calc(23rem - 30px);
}

.notification-dropdown .dropdown-content {
  overflow-y: unset;
}

@media (max-width: 1023px) {
  .header {
    margin: 10px 10px 10px 10px;
  }
  .menu-mobile {
    display: block;
  }
  .menu-desktop {
    display: none;
  }
  .dropdown a.alert-card,
  .dropdown a.alert-card-title {
    width: auto !important;
  }
  .logout,
  .menu-item-with-ico {
    gap: 1rem;
  }

  .text-holder {
    overflow-y: auto;
    max-height: calc(100vh - 80px) !important;
  }
  .notification-dropdown .dropdown-content {
    max-height: calc(100vh - 30px) !important;
    // overflow-y: scroll;
  }
}

.dropdown-content {
  border-radius: 0.75rem;
  box-shadow: 0 0 1rem rgba(0, 0, 0, 0.25);
}

.dropdown-menu {
  min-width: 0rem;
  top: 25px;
}

.dropdown-menu:before {
  content: "";
  position: absolute;
  border-color: rgba(194, 225, 245, 0);
  border: solid transparent;
  border-bottom-color: white;
  border-width: 11px;
  margin-left: -6px;
  top: -15px;
  right: 24px;
  z-index: 1;
}

.alert-icon::before {
  content: attr(data-alerts-count);
  display: block;
  width: 12px;
  min-width: 15px;
  height: 12px;
  min-height: 15px;
  border-radius: 10px;
  background-color: #d23f31;
  color: white;
  font-size: 0.9em;
  top: -6px;
  left: 8px;
  position: absolute;
  text-align: center;
}

.dropdown a.alert-card,
.dropdown a.alert-card-title {
  width: auto;
  word-wrap: break-word;
  display: block;
  white-space: normal;
}

.dropdown a.alert-card::before {
  display: inline-block;
  font: normal normal normal 18px/1 "Material Design Icons";
  content: "󰂜";
  left: -25px;
}

.dropdown a.dismiss-alert-card::after {
  display: inline-block;
  font-family: "Material Icons";
  content: "\e877";
  margin-left: 20px;
}

@media screen and (max-width: 768px) {
  .media-content {
    overflow-x: hidden;
  }
}

@media screen and (min-width: 1023px) {
  .dropdown a.alert-card,
  .dropdown a.alert-card-title {
    width: 350px !important;
  }
}
</style>
<style lang="scss">
.dropdown-menu {
  min-width: auto;
}

.logout,
i.mdi.mdi-logout {
  color: #e63148;
}

@media (max-width: 1023px) {
  .info-opt .dropdown-content {
    max-width: 65vw;
  }
}

p.initials {
  border-radius: 100%;
  text-align: center;
  background-color: #dbdbdb;
  font-size: 1.25rem;
  cursor: default;
  user-select: none;
  display: table-cell;
  vertical-align: middle;
  text-transform: uppercase;
  color: white;
}
</style>
