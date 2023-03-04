<template>
  <div class="sidebar" :class="visibilityClass" id="sidebar">
    <div>
      <img class="img-logo" src="./../assets/logo.png" />
      <div class="txt-logo">
        <span>AItomatic s.l.</span><br /><span>AI Trader</span>
      </div>
    </div>
    <b-menu>
      <b-menu-list>
        <b-menu-item
          icon="area-chart"
          size="is-small"
          :label="expandSB ? 'Live' : ''"
          @click="routeToLive"
        />
        <b-menu-item
          icon="database"
          size="is-small"
          :label="expandSB ? 'History' : ''"
          @click="routeToHistory"
        />
        <b-menu-item
          icon="money"
          :label="expandSB ? 'Results' : ''"
          @click="routeToResults"
        >
        </b-menu-item>
      </b-menu-list>
    </b-menu>
  </div>
</template>

<script lang="ts">
import { Component, Prop, Vue } from "vue-property-decorator";
import { Routes } from "@/router";

@Component({})
export default class Sidebar extends Vue {
  @Prop({ default: true })
  public expandSB!: boolean;

  @Prop({ default: false })
  public showSB!: boolean;

  public get visibilityClass(): string {
    let classes = "";
    if (!this.showSB) {
      classes += "hide";
    }
    if (!this.expandSB) {
      classes += " sidebar-reduced";
    }
    return classes;
  }

  private routeToLive() {
    if (this.$route.name !== Routes.Live) {
      this.$router.push({ name: Routes.Live });
    }
  }

  private routeToHistory() {
    if (this.$route.name !== Routes.History) {
      this.$router.push({ name: Routes.History });
    }
  }

  private routeToResults() {
    if (this.$route.name !== Routes.Results) {
      this.$router.push({ name: Routes.Results });
    }
  }
}
</script>

<style lang="scss">
.sidebar {
  grid-area: sidebar;
  width: 243px;
  transition: width 0.5s;
  height: 100%;
  background: #edeced;
  display: flex;
  flex-direction: column;
}

.logo {
  text-align: left;
  height: 4.37rem;
}

.img-logo {
  margin-top: 30px;
  width: 3rem;
  height: 3rem;
  border-radius: 0.75rem;
  margin-left: 34px;
}
.txt-logo {
  margin-top: 2.56rem;
  float: right;
  margin-right: 55px;
  line-height: 0.84rem;
  color: rgb(87, 87, 87);
}

.txt-logo span:nth-child(1) {
  font-size: 0.74rem;
  font-weight: bold;
}
.txt-logo span:nth-last-child(1) {
  font-size: 1.12rem;
}

.mdi {
  font-size: 1.12rem;
  color: rgba(87, 87, 87, 0.85);
  cursor: pointer;
}

.menu-news {
  a,
  span > i {
    color: #f49600 !important;
  }
}

.no-display {
  display: none;
}

.menu {
  height: 100vh;
  display: flex;
  flex-direction: column;
  overflow: auto;
  margin: 70px 0 0 0;

  > ul:first-child {
    flex-grow: 1;
  }
}

.menu > ul:nth-last-child(2) {
  margin-top: auto;
}

.menu > ul:nth-last-child(1) {
  border-top: 1px solid rgba(231, 231, 231, 0.95);
  padding: 10px 0;
  margin-left: 10px;
}

.menu > ul:nth-last-child(2) a {
  margin-left: -10px;
}

.menu > ul:nth-last-child(1) a {
  margin-left: -10px;
}
.menu > ul:nth-last-child(1) li a {
  white-space: pre-wrap;
}

.menu ul li a .icon {
  margin-left: 13px;
  margin-right: 10px;
}
.menu-list li a span:nth-child(2) {
  font-size: 14px !important;
}

.sidebar-reduced {
  width: 80px;

  .txt-logo {
    display: none;
  }
  .img-logo {
    margin-left: 1rem;
    width: 3rem;
    height: 3rem;
    border-radius: 0.75rem;
  }
  .menu > ul:nth-last-child(1) li:nth-last-child(2) a {
    // color: #f49600 !important;
    white-space: pre-wrap;
  }
  .menu ul li a .icon {
    margin-left: 0px;
    margin-right: 0px;
  }
  .menu-list {
    li {
      a {
        text-align: center !important;
      }
    }
  }
}

@media (max-width: 1023px) {
  .hide {
    display: none; // none
    position: absolute;
    z-index: 3;
    transition: width 0.5s;
  }

  .menu {
    height: 100vh;
  }
}
</style>
