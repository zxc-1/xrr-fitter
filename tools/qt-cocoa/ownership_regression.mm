// Native regression test for Qt 6.11.2 Cocoa accessibility ownership.
// This uses native objects in-process, not desktop/UI automation.
#include <AppKit/AppKit.h>
#include <QtWidgets/QApplication>
#include <QtWidgets/QTableWidget>
#include <QtGui/qaccessible.h>
#include "qcocoaaccessibilityelement.h"
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <dlfcn.h>
#include <objc/runtime.h>

static void require(bool condition, const char *message)
{
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message);
        std::fflush(stderr);
        std::_Exit(1);
    }
}

struct Fixture {
    std::unique_ptr<QTableWidget> widget = std::make_unique<QTableWidget>(2, 2);
    QAccessibleInterface *iface = nullptr;
    QAccessible::Id tableId = 0;
    id native = nil;
    Class nativeClass = Nil;

    Fixture() {
        widget->setItem(0, 0, new QTableWidgetItem("before"));
        widget->setItem(1, 1, new QTableWidgetItem("second"));
        iface = QAccessible::queryAccessibleInterface(widget.get());
        require(iface && iface->tableInterface(), "real Qt table interface exists");
        tableId = QAccessible::uniqueId(iface);
        nativeClass = NSClassFromString(@"QMacAccessibilityElement");
        require(nativeClass != Nil, "Cocoa platform class loaded");
        native = [[nativeClass elementWithInterface:iface] retain];
        require([[native valueForKey:@"rows"] count] == 2, "native rows populated");
    }

    void checkAlive() const {
        require(QAccessible::accessibleInterface(tableId) == iface,
                "live table interface survives native snapshot retirement");
        require(iface->isValid(), "live table interface remains valid");
    }

    ~Fixture() {
        widget.reset();
        require(QAccessible::accessibleInterface(tableId) == nullptr,
                "Qt table interface is released with model owner");
        require([native qtInterface] == nullptr, "native table is invalid after owner destruction");
        [native release];
    }
};

static void borrowedDestructor()
{
    Fixture f;
    @autoreleasepool {
        id row = [[f.nativeClass alloc] initWithId:f.tableId role:NSAccessibilityRowRole];
        [row release];
    }
    f.checkAlive();
}

static void snapshotInvalidation()
{
    Fixture f;
    id oldRow = [[[f.native valueForKey:@"rows"] objectAtIndex:0] retain];
    id oldColumn = [[[f.native valueForKey:@"columns"] objectAtIndex:0] retain];
    @autoreleasepool {
        [f.native updateTableModel];
        require([oldRow qtInterface] == nullptr, "retired row cannot resolve replacement model");
        require([oldColumn qtInterface] == nullptr, "retired column cannot resolve replacement model");
        require([oldRow accessibilityChildren] == nil, "retired row has no accessible children");
    }
    f.checkAlive();
    require([[f.native valueForKey:@"rows"] objectAtIndex:0] != oldRow,
            "refreshed native row is a new snapshot");
    [oldRow release];
    [oldColumn release];
}

static void placeholderInvalidation()
{
    Fixture f;
    id oldRow = [[[f.native valueForKey:@"rows"] objectAtIndex:0] retain];
    (void)[oldRow accessibilityChildren];
    id placeholder = [[[oldRow valueForKey:@"columns"] objectAtIndex:1] retain];
    require([placeholder valueForKey:@"synthesizedRole"] == NSAccessibilityCellRole,
            "test retains a still-unmaterialized cell placeholder");
    @autoreleasepool {
        [f.native updateTableModel];
        require([placeholder qtInterface] == nullptr,
                "retired placeholder cannot create an interface in the replacement model");
    }
    f.checkAlive();
    [placeholder release];
    [oldRow release];
}

static void materializedCell()
{
    Fixture f;
    auto *cell = f.iface->tableInterface()->cellAt(0, 0);
    const auto cellId = QAccessible::uniqueId(cell);
    id nativeCell = [[f.nativeClass elementWithInterface:cell] retain];
    @autoreleasepool {
        [f.native updateTableModel];
    }
    f.checkAlive();
    require(QAccessible::accessibleInterface(cellId) == cell,
            "real cell interface survives old native row deallocation");
    f.widget->item(0, 0)->setText("after");
    require([nativeCell qtInterface] == cell, "shared native cell remains connected");
    require(cell->text(QAccessible::Name) == "after", "shared cell observes current model data");
    f.widget.reset();
    require(QAccessible::accessibleInterface(cellId) == nullptr,
            "real cell interface is released when Qt model is destroyed");
    require([nativeCell qtInterface] == nullptr, "retained native cell invalidates on owner destruction");
    [nativeCell release];
}

static void nativeInvalidation()
{
    Fixture f;
    id oldRow = [[[f.native valueForKey:@"rows"] objectAtIndex:0] retain];
    @autoreleasepool {
        [f.native performSelector:@selector(invalidate)];
    }
    f.checkAlive();
    require([oldRow qtInterface] == nullptr, "invalidated table retires its borrowed rows");
    require([f.native qtInterface] == nullptr, "native table has no interface after invalidation");
    [oldRow release];
}

int main(int argc, char **argv)
{
    QApplication app(argc, argv);
    require(argc == 2, "pass exactly one test name");
    Class nativeClass = NSClassFromString(@"QMacAccessibilityElement");
    require(nativeClass != Nil, "Cocoa platform class loaded");
    Method method = class_getInstanceMethod(nativeClass, @selector(updateTableModel));
    Dl_info origin{};
    require(dladdr(reinterpret_cast<void *>(method_getImplementation(method)), &origin),
            "loaded plugin identity resolves");
    std::printf("PLUGIN: %s\n", origin.dli_fname);
    std::fflush(stdout);
    const QString test = QString::fromLocal8Bit(argv[1]);
    @autoreleasepool {
        if (test == "borrowed-destructor") borrowedDestructor();
        else if (test == "snapshot-invalidation") snapshotInvalidation();
        else if (test == "placeholder-invalidation") placeholderInvalidation();
        else if (test == "materialized-cell") materializedCell();
        else if (test == "native-invalidation") nativeInvalidation();
        else if (test == "repeated-lifecycle") {
            for (int i = 0; i < 100; ++i) {
                @autoreleasepool {
                    borrowedDestructor();
                    snapshotInvalidation();
                    placeholderInvalidation();
                    materializedCell();
                    nativeInvalidation();
                }
            }
        } else require(false, "unknown test name");
    }
    std::printf("PASS: %s\n", argv[1]);
    return 0;
}
