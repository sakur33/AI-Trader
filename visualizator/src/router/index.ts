/* eslint-disable no-unused-vars, no-shadow */

import Vue from "vue";
import VueRouter, { RouteConfig } from "vue-router";
import HistoricCharts from "@/views/HistoricCharts.vue";
import LiveStream from "@/views/LiveStream.vue";
import TradingResults from "@/views/TradingResults.vue";

Vue.use(VueRouter);

export enum Routes {
  Live = "live",
  History = "history",
  Results = "results",
}

const routes: Array<RouteConfig> = [
  {
    // redirection of / -> /jobs
    path: "/",
    redirect: {
      name: Routes.Live,
    },
  },
  {
    // /logs to logsView
    path: "/" + Routes.Live,
    name: Routes.Live,
    component: LiveStream,
  },
  {
    // /system to SystemView
    path: "/" + Routes.History,
    name: Routes.History,
    component: HistoricCharts,
  },
  {
    // /config to ConfigView
    path: "/" + Routes.Results,
    name: Routes.Results,
    component: TradingResults,
  },
];

const router = new VueRouter({
  // fools the browser into thinking that it is changing the path when routing.
  // We are actually a single page WP!
  mode: "history",
  routes,
});

export default router;
