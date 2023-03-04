import Vue from "vue";
import Buefy, { ConfigProgrammatic } from "buefy";
import "./plugins/ApexCharts";
import App from "./App.vue";
import router from "./router";
import "font-awesome/css/font-awesome.min.css";

Vue.config.productionTip = false;

Vue.use(Buefy);

ConfigProgrammatic.setOptions({
  defaultIconPack: "fa",
  // customIconPacks: {
  //   fas: {
  //     default: "lg",
  //     "is-small": "1x",
  //     "is-medium": "2x",
  //     "is-large": "3x",
  //   },
  // },
});

new Vue({
  router,
  render: (h) => h(App),
}).$mount("#app");
