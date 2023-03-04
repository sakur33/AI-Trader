<template>
  <div class="tile is-ancestor">
    <div class="tile is-3 is-vertical is-parent">
      <div class="tile is-child box">
        <div class="panel">
          <p class="panel-heading">Control Panel</p>
          <div class="panel-block">
            <!-- Control Panel content here -->
          </div>
        </div>
      </div>
      <div class="tile is-child box">
        <div class="panel">
          <p class="panel-heading">Data Table</p>
          <b-table :data="tableData" :columns="tableColumns"></b-table>
        </div>
      </div>
    </div>
    <div class="tile is-parent">
      <div class="tile is-child box">
        <v-chart :option="option" autoresize />
      </div>
    </div>
  </div>
</template>

<script>
import { use } from "echarts/core";
import { CanvasRenderer } from "echarts/renderers";
import { PieChart, CandlestickChart } from "echarts/charts";
import {
  TitleComponent,
  TooltipComponent,
  LegendComponent,
} from "echarts/components";
import VChart, { THEME_KEY } from "vue-echarts";

use([
  CanvasRenderer,
  PieChart,
  CandlestickChart,
  TitleComponent,
  TooltipComponent,
  LegendComponent,
]);

export default {
  components: {
    VChart,
  },
  provide: {
    [THEME_KEY]: "dark",
  },
  data() {
    return {
      option: {
        xAxis: {
          data: ["2017-10-24", "2017-10-25", "2017-10-26", "2017-10-27"],
        },
        yAxis: {},
        series: [
          {
            type: "pie",
            data: [
              [20, 34, 10, 38],
              [40, 35, 30, 50],
              [31, 38, 33, 44],
              [38, 15, 5, 42],
            ],
          },
        ],
      },
      tableData: [
        {
          id: 1,
          first_name: "Jesse",
          last_name: "Simmons",
          date: "2016-10-15 13:43:27",
          gender: "Male",
        },
        {
          id: 2,
          first_name: "John",
          last_name: "Jacobs",
          date: "2016-12-15 06:00:53",
          gender: "Male",
        },
        {
          id: 3,
          first_name: "Tina",
          last_name: "Gilbert",
          date: "2016-04-26 06:26:28",
          gender: "Female",
        },
        {
          id: 4,
          first_name: "Clarence",
          last_name: "Flores",
          date: "2016-04-10 10:28:46",
          gender: "Male",
        },
        {
          id: 5,
          first_name: "Anne",
          last_name: "Lee",
          date: "2016-12-06 14:38:38",
          gender: "Female",
        },
      ],
      tableColumns: [
        {
          field: "id",
          label: "ID",
          width: "40",
          numeric: true,
        },
        {
          field: "first_name",
          label: "First Name",
        },
        {
          field: "last_name",
          label: "Last Name",
        },
        {
          field: "date",
          label: "Date",
          centered: true,
        },
        {
          field: "gender",
          label: "Gender",
        },
      ],
    };
  },
};
</script>
