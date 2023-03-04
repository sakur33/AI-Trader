<template>
  <div class="tile is-ancestor">
    <div class="tile is-4 is-vertical is-parent">
      <div class="tile is-child box">
        <div class="panel">
          <p class="panel-heading">Control Panel</p>
          <div class="panel-block">
            <b-dropdown aria-role="list">
              <template #trigger="{ active }">
                <b-button
                  label="Click me!"
                  type="is-primary"
                  :icon-right="active ? 'menu-up' : 'menu-down'"
                />
              </template>

              <b-dropdown-item aria-role="listitem">Action</b-dropdown-item>
              <b-dropdown-item aria-role="listitem"
                >Another action</b-dropdown-item
              >
              <b-dropdown-item aria-role="listitem"
                >Something else</b-dropdown-item
              >
            </b-dropdown>
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
        <apexChart type="candlestick" :options="options" :series="series">
        </apexChart>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  components: {
    // VChart,
    // apexCharts,
  },
  data() {
    return {
      options: {
        chart: {
          id: "vuechart-example",
          height: "100%",
          width: "100%",
          redrawOnParentResize: true,
        },
      },
      series: [
        {
          data: [
            {
              x: new Date("2016-1-1").toLocaleString(),
              y: [51.98, 56.29, 51.59, 53.85],
            },
            {
              x: new Date(2016, 2, 1).toLocaleString(),
              y: [53.66, 54.99, 51.35, 52.95],
            },
            {
              x: new Date(2016, 3, 1).toLocaleString(),
              y: [52.76, 57.35, 52.15, 57.03],
            },
            {
              x: new Date(2016, 4, 1).toLocaleString(),
              y: [52.76, 57.35, 52.15, 57.03],
            },
            {
              x: new Date(2016, 5, 1).toLocaleString(),
              y: [52.76, 57.35, 52.15, 57.03],
            },
            {
              x: new Date(2016, 6, 1).toLocaleString(),
              y: [52.76, 57.35, 52.15, 57.03],
            },
          ],
        },
      ],
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
